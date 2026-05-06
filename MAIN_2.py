import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.decomposition import PCA
import pickle
import json
from datetime import datetime
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize


class MDMSeparationSolver:
    def __init__(self, scheme: str = 'cyclic', tol: float = 1e-8, max_iter: int = 50000):
        self.scheme = scheme.lower()
        self.tol = tol
        self.max_iter = max_iter

    @staticmethod
    def _delta_k(dots_k, lam_k, tol=1e-6):
        active = lam_k > tol
        if not np.any(active):
            return 0.0
        max_active = dots_k[active].max()
        min_all = dots_k.min()
        return max(0.0, max_active - min_all)

    def fit(self, P_list):
        s = len(P_list)
        dk = len(P_list[0])
        n = len(P_list[0][0])
        P = np.array(P_list, dtype=np.float64)
        lambdas = np.ones((s, dk), dtype=np.float64) / dk
        v = np.zeros(n, dtype=np.float64)
        for k in range(s):
            v += lambdas[k] @ P[k]

        consecutive_zeros = 0

        for it in range(self.max_iter):
            dots = P @ v

            if self.scheme == 'cyclic':
                for k_idx in range(s):
                    delta_k = self._delta_k(dots[k_idx], lambdas[k_idx], self.tol)

                    if delta_k > self.tol:
                        if self._step_correct(k_idx, P[k_idx], dots[k_idx], lambdas[k_idx], v, self.tol):
                            consecutive_zeros = 0
                            dots = P @ v
                        else:
                            consecutive_zeros += 1
                    else:
                        consecutive_zeros += 1

                    if consecutive_zeros >= s:
                        break
            else:
                deltas = np.array([self._delta_k(dots[k], lambdas[k], self.tol) for k in range(s)])
                if deltas.max() > self.tol:
                    k_max = int(deltas.argmax())
                    success = self._step_correct(k_max, P[k_max], dots[k_max], lambdas[k_max], v, self.tol)
                    if success:
                        consecutive_zeros = 0
                        dots = P @ v
                    else:
                        consecutive_zeros += 1
                else:
                    consecutive_zeros += 1

            if all(self._delta_k(dots[k], lambdas[k], self.tol) < self.tol for k in range(s)):
                break

        final_norm = float(np.linalg.norm(v))
        return {
            'v_star': v,
            'separation_distance': final_norm,
            'converged': it < self.max_iter - 1,
            'iterations': it + 1
        }

    @staticmethod
    def _step_correct(k, Pk, dots_k, lam_k, v, tol=1e-15):
        active_idx = np.where(lam_k > tol)[0]
        if len(active_idx) == 0:
            return False

        i_prime = active_idx[np.argmax(dots_k[active_idx])]
        i_pp = np.argmin(dots_k)

        delta_k = dots_k[i_prime] - dots_k[i_pp]
        if delta_k < tol:
            return False

        diff = Pk[i_prime] - Pk[i_pp]
        diff_norm_sq = np.dot(diff, diff)
        if diff_norm_sq < tol:
            return False

        t_star = min(lam_k[i_prime], delta_k / diff_norm_sq)
        if t_star <= tol:
            return False

        v -= t_star * diff
        lam_k[i_prime] -= t_star
        lam_k[i_pp] += t_star

        return True


class MDMMultiClassifier:
    def __init__(self):
        self.separators = {}
        self.class_names = []
        self.use_pca = False
        self.pca = None
        self.pca_components = 128
        self.model_info = {}

    def save_model(self, filepath):
        model_data = {
            'separators': self.separators,
            'class_names': self.class_names,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'pca': self.pca,
            'model_info': self.model_info,
            'saved_at': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file {filepath} not found")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.separators = model_data['separators']
        self.class_names = model_data['class_names']
        self.use_pca = model_data['use_pca']
        self.pca_components = model_data['pca_components']
        self.pca = model_data['pca']
        self.model_info = model_data.get('model_info', {})

        print(f"Model loaded from {filepath}")
        print(f"   Classes: {', '.join(self.class_names)}")
        print(f"   Saved: {model_data.get('saved_at', 'unknown')}")
        return self

    def train(self, features_dict, use_pca=False, pca_components=128):
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.class_names = list(features_dict.keys())
        self.model_info = {
            'trained_at': datetime.now().isoformat(),
            'n_classes': len(self.class_names),
            'features_per_class': {cls: len(feats) for cls, feats in features_dict.items()},
            'use_pca': use_pca,
            'pca_components': pca_components if use_pca else None
        }

        print(f"Training {len(self.class_names)} OvR MDM models with calibration...")

        if use_pca:
            print(f"  Applying PCA to reduce dimension to {pca_components}...")
            all_features = np.vstack(list(features_dict.values()))
            self.pca = PCA(n_components=pca_components, whiten=True, random_state=42)
            self.pca.fit(all_features)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"  PCA explains {explained_var:.1%} variance")
            self.model_info['pca_explained_variance'] = float(explained_var)

        for cls in self.class_names:
            print(f"  '{cls}' vs others...")
            pos = features_dict[cls]
            neg = np.vstack([features_dict[o] for o in self.class_names if o != cls])

            if use_pca and self.pca is not None:
                pos = self.pca.transform(pos)
                neg = self.pca.transform(neg)

            min_len = min(len(pos), len(neg))
            pos = pos[np.random.choice(len(pos), min_len, replace=False)]
            neg = neg[np.random.choice(len(neg), min_len, replace=False)]

            pos = normalize(pos, norm='l2')
            neg = normalize(neg, norm='l2')

            solver = MDMSeparationSolver(scheme='cyclic', tol=1e-8)
            res = solver.fit([pos.tolist(), neg.tolist()])

            v = res['v_star']
            dist = res['separation_distance']

            pos_scores = pos @ v
            neg_scores = neg @ v
            if np.mean(pos_scores) < np.mean(neg_scores):
                v = -v
                pos_scores = -pos_scores
                neg_scores = -neg_scores

            threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2.0

            train_preds = (pos_scores > threshold).astype(int) + (neg_scores > threshold).astype(int) * 0
            acc = np.mean(train_preds == 1)
            acc_neg = np.mean((neg_scores <= threshold).astype(int) == 1)
            avg_acc = (acc + acc_neg) / 2

            self.separators[cls] = {'v': v, 'threshold': threshold, 'dist': dist}
            print(f"  Dist: {dist:.4f} | Threshold: {threshold:.4f} | Train Acc: {avg_acc:.1%}")

            if avg_acc < 0.75:
                print(f"  Warning: '{cls}' is not well separated in this feature space!")

        self.model_info['train_accuracy'] = {
            cls: float(self.separators[cls]['dist']) for cls in self.class_names
        }
        print("Training and calibration completed.\n")

    def predict(self, image_path, return_details=False):
        feat = extract_enhanced_features_single(image_path, use_pca=self.use_pca,
                                                   pca=self.pca if self.use_pca else None)
        feat = normalize(feat.reshape(1, -1), norm='l2').flatten()

        scores = {}
        for cls, params in self.separators.items():
            raw = np.dot(feat, params['v'])
            scores[cls] = (raw - params['threshold']) / (params['dist'] + 1e-8)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        pred = sorted_scores[0][0]
        margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        confidence = 1.0 / (1.0 + np.exp(-2.0 * margin))

        if return_details:
            return pred, confidence, scores
        return pred, confidence, scores


class ImprovedMDMMultiClassifier(MDMMultiClassifier):
    def __init__(self):
        super().__init__()
        self.scaler = None

    def save_model(self, filepath):
        model_data = {
            'separators': self.separators,
            'class_names': self.class_names,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'pca': self.pca,
            'scaler': self.scaler,
            'model_info': self.model_info,
            'saved_at': datetime.now().isoformat(),
            'model_type': 'ImprovedMDMMultiClassifier'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Improved model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file {filepath} not found")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.separators = model_data['separators']
        self.class_names = model_data['class_names']
        self.use_pca = model_data['use_pca']
        self.pca_components = model_data['pca_components']
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.model_info = model_data.get('model_info', {})

        print(f"Improved model loaded from {filepath}")
        print(f"   Classes: {', '.join(self.class_names)}")
        print(f"   Saved: {model_data.get('saved_at', 'unknown')}")
        if 'model_info' in model_data and 'trained_at' in model_data['model_info']:
            print(f"   Trained: {model_data['model_info']['trained_at']}")
        return self

    def train(self, features_dict, use_pca=False, pca_components=32, use_scaling=True):
        self.class_names = list(features_dict.keys())
        self.use_pca = use_pca
        self.pca_components = pca_components if use_pca else None

        self.model_info = {
            'trained_at': datetime.now().isoformat(),
            'n_classes': len(self.class_names),
            'features_per_class': {cls: len(feats) for cls, feats in features_dict.items()},
            'use_pca': use_pca,
            'pca_components': pca_components if use_pca else None,
            'use_scaling': use_scaling
        }

        print(f"Training {len(self.class_names)} OvR MDM models...")

        if use_scaling:
            print("  Applying RobustScaler to features...")
            all_features = np.vstack(list(features_dict.values()))
            self.scaler = RobustScaler(quantile_range=(5, 95))
            self.scaler.fit(all_features)

            scaled_dict = {}
            for cls, feats in features_dict.items():
                scaled_dict[cls] = self.scaler.transform(feats)
            features_dict = scaled_dict

        if use_pca:
            print(f"  Applying PCA to reduce dimension to {pca_components}...")
            all_features = np.vstack(list(features_dict.values()))
            self.pca = PCA(n_components=pca_components, whiten=True, random_state=42)
            self.pca.fit(all_features)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"  PCA explains {explained_var:.1%} variance")
            self.model_info['pca_explained_variance'] = float(explained_var)

            pca_dict = {}
            for cls, feats in features_dict.items():
                pca_dict[cls] = self.pca.transform(feats)
            features_dict = pca_dict

        for cls in self.class_names:
            print(f"  '{cls}' vs others...")
            pos = features_dict[cls]
            neg = np.vstack([features_dict[o] for o in self.class_names if o != cls])

            min_len = min(len(pos), len(neg), 150)
            if len(pos) > min_len:
                pos = pos[np.random.choice(len(pos), min_len, replace=False)]
            if len(neg) > min_len:
                neg = neg[np.random.choice(len(neg), min_len, replace=False)]

            pos = normalize(pos, norm='l2')
            neg = normalize(neg, norm='l2')

            pos += np.random.normal(0, 1e-6, pos.shape)
            neg += np.random.normal(0, 1e-6, neg.shape)

            solver = MDMSeparationSolver(scheme='cyclic', tol=1e-6, max_iter=10000)
            res = solver.fit([pos.tolist(), neg.tolist()])

            v = res['v_star']
            dist = res['separation_distance']

            if dist < 1e-4:
                pos_mean = pos.mean(axis=0)
                neg_mean = neg.mean(axis=0)
                v = (pos_mean - neg_mean)
                v = v / (np.linalg.norm(v) + 1e-8)
                dist = np.linalg.norm(pos_mean - neg_mean)

            pos_scores = pos @ v
            neg_scores = neg @ v

            if np.mean(pos_scores) < np.mean(neg_scores):
                v = -v
                pos_scores = -pos_scores
                neg_scores = -neg_scores

            threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2.0

            pos_acc = np.mean(pos_scores > threshold)
            neg_acc = np.mean(neg_scores <= threshold)
            avg_acc = (pos_acc + neg_acc) / 2

            self.separators[cls] = {'v': v, 'threshold': threshold, 'dist': dist}
            print(f"  Dist: {dist:.4f} | Threshold: {threshold:.4f} | Acc: {avg_acc:.1%}")

        self.model_info['train_accuracy'] = {
            cls: float(self.separators[cls]['dist']) for cls in self.class_names
        }
        print("Training completed.\n")

    def predict(self, image_path, return_details=False):
        feat = extract_enhanced_features_single(image_path)

        if self.scaler is not None:
            feat = self.scaler.transform(feat.reshape(1, -1)).flatten()

        if self.pca is not None:
            feat = self.pca.transform(feat.reshape(1, -1)).flatten()

        feat = normalize(feat.reshape(1, -1), norm='l2').flatten()

        scores = {}
        for cls, params in self.separators.items():
            raw = np.dot(feat, params['v'])
            scores[cls] = (raw - params['threshold']) / (params['dist'] + 1e-8)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        pred = sorted_scores[0][0]
        margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        confidence = 1.0 / (1.0 + np.exp(-2.0 * margin))

        if return_details:
            return pred, confidence, scores
        return pred, confidence, scores


def extract_enhanced_features_single(image_path, img_size=(128, 128), use_pca=False, pca=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    color_moments = []
    for channel in cv2.split(hsv):
        color_moments.extend([
            np.mean(channel),
            np.std(channel),
            np.mean(np.abs(channel - np.mean(channel)) ** 3) ** (1. / 3)
        ])

    h_hist = cv2.normalize(h_hist, h_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    s_hist = cv2.normalize(s_hist, s_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    v_hist = cv2.normalize(v_hist, v_hist, alpha=1.0, norm_type=cv2.NORM_L1)

    color_feat = np.concatenate([h_hist, s_hist, v_hist, color_moments])

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    from skimage.feature import graycomatrix, graycoprops

    glcm_img = resize(gray, (64, 64), preserve_range=True).astype(np.uint8)
    glcm = graycomatrix(glcm_img, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)

    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        glcm_features.extend(graycoprops(glcm, prop).flatten())

    hog_feat = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                   block_norm='L2-Hys', visualize=False, transform_sqrt=True,
                   feature_vector=True)

    sift = cv2.SIFT_create(
        nfeatures=200,
        nOctaveLayers=3,
        contrastThreshold=0.03,
        edgeThreshold=15,
        sigma=1.6
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is not None and len(descriptors) > 0:
        sift_mean = np.mean(descriptors, axis=0)
        sift_std = np.std(descriptors, axis=0)
        sift_max = np.max(descriptors, axis=0)
        sift_min = np.min(descriptors, axis=0)
        sift_feat = np.concatenate([sift_mean, sift_std, sift_max, sift_min])

        n_keypoints = min(len(keypoints) / 1000, 1.0)
        sift_feat = np.concatenate([[n_keypoints], sift_feat])
    else:
        sift_feat = np.zeros(1 + 128 * 4)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_features = []
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

        shape_features = [circularity, area / (img_size[0] * img_size[1])]
        shape_features.extend(hu_moments[:4])
    else:
        shape_features = np.zeros(6)

    dct = cv2.dct(np.float32(gray) / 255.0)
    dct_low = dct[:16, :16].flatten()

    combined_features = np.concatenate([
        color_feat,
        lbp_hist,
        glcm_features,
        hog_feat,
        sift_feat,
        shape_features,
        dct_low
    ])

    if use_pca and pca is not None:
        combined_features = pca.transform(combined_features.reshape(1, -1)).flatten()

    combined_features = cv2.normalize(combined_features, combined_features,
                                      alpha=1.0, norm_type=cv2.NORM_L2)

    return combined_features


def extract_enhanced_features_folder(folder_path, img_size=(128, 128), verbose=True):
    folder = Path(folder_path)
    image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    features = []

    for i, img_path in enumerate(image_paths):
        try:
            feat = extract_enhanced_features_single(img_path, img_size)
            features.append(feat)
            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1} of {len(image_paths)} images in {folder_path}")
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")

    if verbose:
        print(f"  Extracted features: {len(features)} of {len(image_paths)}")
        if features:
            print(f"  Feature dimension: {features[0].shape[0]}")

    return np.array(features, dtype=np.float64)


if __name__ == "__main__":
    MODEL_PATH = "flower_classifier_3flowers.pkl"

    if Path(MODEL_PATH).exists():
        print("=" * 50)
        print("Found saved model!")
        print("=" * 50)
        response = input("Load existing model? (y/n): ").strip().lower()

        while response not in ['y', 'n', 'yes', 'no']:
            print("Please enter 'y' (yes) or 'n' (no)")
            response = input("Load existing model? (y/n): ").strip().lower()

        response = 'y' if response in ['y', 'yes'] else 'n'

        if response == 'y':
            print("\nLoading model...")
            clf = ImprovedMDMMultiClassifier()
            try:
                clf.load_model(MODEL_PATH)
                print("\nModel loaded successfully!")

                print("\n=== TESTING ===")
                test_image_path = "test_flower.jpg"

                if not Path(test_image_path).exists():
                    test_image_path = input("Path to test image: ").strip()

                if Path(test_image_path).exists():
                    pred_class, confidence, all_scores = clf.predict(test_image_path, return_details=True)
                    print(f"\nImage: {test_image_path}")
                    print(f"Predicted class: {pred_class.upper()}")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"\nClass scores:")
                    for cls, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {cls}: {score:.4f}")

                    img_display = cv2.imread(test_image_path)
                    if img_display is not None:
                        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                        plt.figure(figsize=(8, 6))
                        plt.imshow(img_display)
                        plt.title(f"Prediction: {pred_class.upper()}\nConfidence: {confidence:.2%}", fontsize=14)
                        plt.axis('off')
                        plt.show()
                else:
                    print(f"Test image {test_image_path} not found")

                exit(0)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training new model...")

    print("\n" + "=" * 50)
    print("STARTING NEW MODEL TRAINING (3 CLASSES)")
    print("=" * 50)
    print("Extracting features: COLOR (HSV) + HOG + SIFT + LBP + GLCM\n")

    classes_config = {
        "roses": "flowers/roses",
        "dandelions": "flowers/dandelions",
        "sunflowers": "flowers/sunflowers"
    }

    print("CHECKING IMAGE FOLDERS:")
    for name, folder in classes_config.items():
        if Path(folder).exists():
            print(f"  {name}: {folder} - found")
        else:
            print(f"  {name}: {folder} - NOT FOUND!")
            print(f"     Create folder '{folder}' and add images of {name}")
            exit(1)

    features_dict = {}
    for name, folder in classes_config.items():
        print(f"\nLoading {name} from {folder}...")
        feats = extract_enhanced_features_folder(folder)

        if len(feats) == 0:
            print(f"No images in folder {folder} for class '{name}'!")
            exit(1)

        min_samples = 200
        if len(feats) > min_samples:
            idx = np.random.choice(len(feats), min_samples, replace=False)
            feats = feats[idx]
            print(f"  Selected {min_samples} images for balancing")

        features_dict[name] = feats
        print(f"  Feature dimension: {feats.shape[1]}")
        print(f"  Number of samples: {feats.shape[0]}")

    print("\n" + "=" * 50)
    print("STARTING CLASSIFIER TRAINING")
    print("=" * 50)

    clf = ImprovedMDMMultiClassifier()
    clf.train(features_dict, use_pca=True, pca_components=128, use_scaling=True)

    print("\n" + "=" * 50)
    save_response = input("Save trained model? (y/n): ").strip().lower()
    while save_response not in ['y', 'n', 'yes', 'no']:
        print("Please enter 'y' (yes) or 'n' (no)")
        save_response = input("Save trained model? (y/n): ").strip().lower()

    if save_response in ['y', 'yes']:
        clf.save_model(MODEL_PATH)
        print(f"Model saved as '{MODEL_PATH}'")

    print("\n" + "=" * 50)
    print("MODEL TESTING")
    print("=" * 50)

    test_image_path = "test_sunflower.jpg"

    if not Path(test_image_path).exists():
        print(f"File {test_image_path} not found.")
        test_image_path = input("Path to test image: ").strip()

    if Path(test_image_path).exists():
        try:
            pred_class, confidence, all_scores = clf.predict(test_image_path, return_details=True)
            print(f"\nImage: {test_image_path}")
            print(f"Predicted class: {pred_class.upper()}")
            print(f"Confidence: {confidence:.2%}")
            print(f"\nClass scores:")
            for cls, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                marker = "" if cls == pred_class else " "
                print(f"  {marker} {cls}: {score:.4f}")

            img_display = cv2.imread(test_image_path)
            if img_display is not None:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 8))
                plt.imshow(img_display)

                color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
                plt.title(f"Prediction: {pred_class.upper()}\nConfidence: {confidence:.2%}",
                          fontsize=14, color=color)
                plt.axis('off')
                plt.show()

                plt.figure(figsize=(8, 4))
                classes = list(all_scores.keys())
                scores = list(all_scores.values())
                colors = ['green' if c == pred_class else 'gray' for c in classes]
                plt.bar(classes, scores, color=colors)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title('Class Scores')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"File {test_image_path} not found.")