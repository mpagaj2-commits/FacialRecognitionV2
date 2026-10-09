import os
import logging
import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

photos_directory = "photos"
database_file = "database.txt"
attendance_file = "attendance.csv"
recorded_attendance = set()

known_face_encodings = []
known_face_names = []

if not os.path.exists(photos_directory):
    os.makedirs(photos_directory)
    logging.info("Dossier 'photos' cree.")

def load_or_build_database():
    valid_extensions = (".jpg", ".jpeg", ".png", ".jfif", ".webp")

    # 1. Charger les profils déjà enregistrés
    if os.path.exists(database_file):
        logging.info("Chargement de la base existante...")

        with open(database_file, "r") as file:
            for line in file:
                parts = line.strip().split("|")

                if len(parts) != 2:
                    continue

                filename = parts[0]

                try:
                    encoding = np.array(
                        [float(x) for x in parts[1].split(",")]
                    )
                except ValueError:
                    logging.warning(
                        "Encodage invalide : %s", filename
                    )
                    continue

                if encoding.shape != (128,):
                    logging.warning(
                        "Taille d'encodage incorrecte : %s",
                        filename
                    )
                    continue

                if filename not in known_face_names:
                    known_face_names.append(filename)
                    known_face_encodings.append(encoding)

    # 2. Construire la liste des profils déjà connus
    existing_photos = set(known_face_names)

    # 3. Parcourir les photos à chaque lancement
    for filename in os.listdir(photos_directory):

        if not filename.lower().endswith(valid_extensions):
            continue

        # 4. Ignorer les photos déjà enregistrées
        if filename in existing_photos:
            continue

        filepath = os.path.join(photos_directory, filename)

        try:
            image_data = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image_data)

            if not encodings:
                logging.warning(
                    "Aucun visage detecte : %s", filename
                )
                continue

            # 5. Enregistrer le nouvel encodage
            encoding = encodings[0]

            with open(database_file, "a") as file:
                encoding_string = ",".join(
                    str(value) for value in encoding
                )
                file.write(f"{filename}|{encoding_string}\n")

            # 6. Mettre à jour les listes en mémoire
            known_face_names.append(filename)
            known_face_encodings.append(encoding)
            existing_photos.add(filename)

            logging.info("Nouveau profil ajoute : %s", filename)

        except Exception as error:
            logging.error(
                "Impossible de traiter %s : %s",
                filename, error
            )

    logging.info(
        "Nombre total de profils : %d",
        len(known_face_names)
    )

load_or_build_database()



def mark_attendance(display_name):
    now = datetime.now()

    date_today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    attendance_key = (date_today, display_name)

    # Ne pas enregistrer plusieurs fois la même personne
    # le même jour
    if attendance_key in recorded_attendance:
        return

    file_exists = (
        os.path.exists(attendance_file)
        and os.path.getsize(attendance_file) > 0
    )

    with open(
        attendance_file,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file, delimiter=";")

        if not file_exists:
            writer.writerow(["Date", "Heure", "Nom"])

        writer.writerow([
            date_today,
            time_now,
            display_name
        ])

    recorded_attendance.add(attendance_key)

    logging.info(
        "Presence enregistree : %s a %s",
        display_name, time_now
    )




def load_existing_attendance():
    if not os.path.exists(attendance_file):
        return

    with open(
        attendance_file,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            date = row.get("Date")
            name = row.get("Nom")

            if date and name:
                recorded_attendance.add((date, name))
load_existing_attendance()                



if not known_face_encodings:
    logging.critical("Aucun profil valide dans la base de donnees.")
    exit(1)

video_capture_device = cv2.VideoCapture(0)

if not video_capture_device.isOpened():
    logging.critical("L'acces au peripherique video est refuse.")
    exit(1)

logging.info("Flux video actif. Appuyez sur 'q' pour quitter.")

while True:
    frame_read_success, current_video_frame = video_capture_device.read()

    if not frame_read_success:
        logging.error("Interruption du flux video.")
        break

    small_video_frame = cv2.resize(current_video_frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_video_frame, cv2.COLOR_BGR2RGB)

    detected_face_locations = face_recognition.face_locations(rgb_small_frame)
    detected_face_encodings = face_recognition.face_encodings(rgb_small_frame, detected_face_locations)

    for current_face_encoding, current_face_location in zip(detected_face_encodings, detected_face_locations):
        encoding_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
        
        display_name = "unknown"
        box_color = (0, 0, 255)

        if len(encoding_distances) > 0:
            best_match_index = np.argmin(encoding_distances)
            if encoding_distances[best_match_index] < 0.6:
                display_name = known_face_names[best_match_index]
                box_color = (0, 255, 0)
                mark_attendance(display_name)

        top_position, right_position, bottom_position, left_position = current_face_location
        
        top_position *= 4
        right_position *= 4
        bottom_position *= 4
        left_position *= 4

        cv2.rectangle(
            current_video_frame, 
            (left_position, top_position), 
            (right_position, bottom_position), 
            box_color, 
            2
        )
        
        cv2.putText(
            current_video_frame, 
            display_name, 
            (left_position, top_position - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            box_color, 
            2
        )

    cv2.imshow("Module de Reconnaissance", current_video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture_device.release()
cv2.destroyAllWindows()
logging.info("Execution terminee.")