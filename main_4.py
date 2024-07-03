import cv2
import numpy as np

# Funktion für den Farbtransfer im Lab-Farbraum
def color_transfer_lab(input_img, target_img):
    # Bilder in den Lab-Farbraum konvertieren
    input_img_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0
    target_img_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0

    # Für jeden Farbkanal (L, a, b) separat vorgehen
    for i in range(3):
        # i) Mittelwert des Input subtrahieren
        input_channel_mean = np.mean(input_img_lab[:, :, i])
        input_img_lab[:, :, i] -= input_channel_mean

        # ii) Input durch Standardabweichung teilen, unter Berücksichtigung der Möglichkeit von Nullen
        input_channel_std = np.std(input_img_lab[:, :, i])
        if input_channel_std != 0:
            input_img_lab[:, :, i] /= input_channel_std

        # iii) Input mit der Standardabweichung des Targets multiplizieren, unter Berücksichtigung der Möglichkeit von Nullen
        target_channel_std = np.std(target_img_lab[:, :, i])
        if target_channel_std != 0:
            input_img_lab[:, :, i] *= target_channel_std

        # iv) Mittelwert des Targets zum Input addieren
        target_channel_mean = np.mean(target_img_lab[:, :, i])
        input_img_lab[:, :, i] += target_channel_mean

    # Clipping der Werte, um sicherzustellen, dass sie im gültigen Bereich bleiben
    input_img_lab = np.clip(input_img_lab, 0, 1)

    # Konvertierung zurück in den 8-Bit-Bereich
    output_img_lab = (input_img_lab * 255).astype(np.uint8)

    # Konvertieren des Bildes zurück in den BGR-Farbraum für die Anzeige
    output_img_bgr_lab = cv2.cvtColor(output_img_lab, cv2.COLOR_Lab2BGR)

    return output_img_bgr_lab

# Funktion für den Farbtransfer im HSV-Farbraum
def color_transfer_hsv(input_img, target_img):
    # Bilder in den HSV-Farbraum konvertieren
    input_img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    target_img_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    # Für jeden Farbkanal (Hue, Saturation, Value) separat vorgehen
    for i in range(3):
        # i) Mittelwert des Input subtrahieren
        input_channel_mean = np.mean(input_img_hsv[:, :, i])
        input_img_hsv[:, :, i] -= input_channel_mean

        # ii) Input durch Standardabweichung teilen, unter Berücksichtigung der Möglichkeit von Nullen
        input_channel_std = np.std(input_img_hsv[:, :, i])
        if input_channel_std != 0:
            input_img_hsv[:, :, i] /= input_channel_std

        # iii) Input mit der Standardabweichung des Targets multiplizieren, unter Berücksichtigung der Möglichkeit von Nullen
        target_channel_std = np.std(target_img_hsv[:, :, i])
        if target_channel_std != 0:
            input_img_hsv[:, :, i] *= target_channel_std

        # iv) Mittelwert des Targets zum Input addieren
        target_channel_mean = np.mean(target_img_hsv[:, :, i])
        input_img_hsv[:, :, i] += target_channel_mean

    # Clipping der Werte, um sicherzustellen, dass sie im gültigen Bereich bleiben
    input_img_hsv = np.clip(input_img_hsv, 0, 1)

    # Konvertierung zurück in den 8-Bit-Bereich
    output_img_hsv = (input_img_hsv * 255).astype(np.uint8)

    # Konvertieren des Bildes zurück in den BGR-Farbraum für die Anzeige
    output_img_bgr_hsv = cv2.cvtColor(output_img_hsv, cv2.COLOR_HSV2BGR)

    return output_img_bgr_hsv

# Funktion für den Farbtransfer im RGB-Farbraum
def color_transfer_rgb(input_img, target_img):
    # Bilder in den RGB-Farbraum konvertieren (falls nicht bereits)
    input_img_rgb = input_img.astype(np.float32) / 255.0
    target_img_rgb = target_img.astype(np.float32) / 255.0

    # Für jeden Farbkanal (R, G, B) separat vorgehen
    for i in range(3):
        # i) Mittelwert des Input subtrahieren
        input_channel_mean = np.mean(input_img_rgb[:, :, i])
        input_img_rgb[:, :, i] -= input_channel_mean

        # ii) Input durch Standardabweichung teilen, unter Berücksichtigung der Möglichkeit von Nullen
        input_channel_std = np.std(input_img_rgb[:, :, i])
        if input_channel_std != 0:
            input_img_rgb[:, :, i] /= input_channel_std

        # iii) Input mit der Standardabweichung des Targets multiplizieren, unter Berücksichtigung der Möglichkeit von Nullen
        target_channel_std = np.std(target_img_rgb[:, :, i])
        if target_channel_std != 0:
            input_img_rgb[:, :, i] *= target_channel_std

        # iv) Mittelwert des Targets zum Input addieren
        target_channel_mean = np.mean(target_img_rgb[:, :, i])
        input_img_rgb[:, :, i] += target_channel_mean

    # Clipping der Werte, um sicherzustellen, dass sie im gültigen Bereich bleiben
    input_img_rgb = np.clip(input_img_rgb, 0, 1)

    # Konvertierung zurück in den 8-Bit-Bereich
    output_img_rgb = (input_img_rgb * 255).astype(np.uint8)

    return output_img_rgb

# Laden der Bilder
input_image = cv2.imread('FigSource.png')
target_image = cv2.imread('FigTarget.png')

# Überprüfen, ob die Bilder geladen wurden
if input_image is None or target_image is None:
    print("Ein oder beide Bilder konnten nicht geladen werden.")
else:
    # Farbtransfer für alle Farbräume durchführen
    result_lab = color_transfer_lab(input_image, target_image)
    result_hsv = color_transfer_hsv(input_image, target_image)
    result_rgb = color_transfer_rgb(input_image, target_image)

    # Bilder nebeneinander anzeigen
    combined_lab = np.concatenate((input_image, result_lab), axis=1)
    combined_hsv = np.concatenate((input_image, result_hsv), axis=1)
    combined_rgb = np.concatenate((input_image, result_rgb), axis=1)

    cv2.imshow('Lab Color Transfer', combined_lab)
    cv2.imshow('HSV Color Transfer', combined_hsv)
    cv2.imshow('RGB Color Transfer', combined_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
