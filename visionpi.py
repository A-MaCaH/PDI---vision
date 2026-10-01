#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionPi - Laboratorio de vision por computadora en terminal (Raspberry Pi 4)
=============================================================================
Un solo archivo. Se controla escribiendo comandos en la terminal.

INSTALACION (Raspberry Pi OS 64-bit, Bookworm):
    sudo apt update
    sudo apt install -y python3-opencv opencv-data python3-venv
    python3 -m venv --system-site-packages ~/visionpi-env
    source ~/visionpi-env/bin/activate
    pip install rich                 # (opcional) interfaz mas bonita
    pip install mediapipe            # (opcional) esqueleto rapido
    pip install ultralytics ncnn     # (opcional) YOLO: objetos y pose

USO:
    python3 visionpi.py                    # camara 0 a 640x480
    python3 visionpi.py --camara 1         # otra camara USB
    python3 visionpi.py --camara prueba    # patron sintetico (sin camara)
    python3 visionpi.py --camara video.mp4 # un archivo de video

VER EL VIDEO:
    - Con monitor/escritorio en la Pi: se abre una ventana (tecla q = salir).
    - Por SSH: abre en tu navegador http://IP-DE-LA-PI:8080

Dentro de la app escribe 'ayuda' para ver todos los comandos.
"""

import argparse
import importlib.util
import os
import socket
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import cv2
    import numpy as np
except ImportError:
    print("Falta OpenCV. En la Raspberry: sudo apt install -y python3-opencv")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _console = Console(highlight=False)
except ImportError:
    _console = None

CARPETA = "capturas"
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Utilidades de terminal
# ---------------------------------------------------------------------------
def decir(msg, estilo=None):
    if _console:
        _console.print(msg, style=estilo, markup=False)
    else:
        print(msg)


def tabla(titulo, columnas, filas):
    if _console:
        t = Table(title=titulo, header_style="bold cyan")
        for c in columnas:
            t.add_column(c)
        for f in filas:
            t.add_row(*[Text(str(x)) for x in f])
        _console.print(t)
        return
    print(f"\n== {titulo} ==")
    anchos = [max([len(str(c))] + [len(str(f[i])) for f in filas])
              for i, c in enumerate(columnas)]
    print("  ".join(str(c).ljust(a) for c, a in zip(columnas, anchos)))
    for f in filas:
        print("  ".join(str(x).ljust(a) for x, a in zip(f, anchos)))


def tiene(modulo):
    return importlib.util.find_spec(modulo) is not None


def ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Filtros (todos reciben y devuelven una imagen BGR del mismo tamano)
# ---------------------------------------------------------------------------
def _a_bgr(gris):
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _gris(img):
    return _a_bgr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def _bordes(img):
    return _a_bgr(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160))


def _sobel(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.convertScaleAbs(cv2.Sobel(g, cv2.CV_16S, 1, 0))
    gy = cv2.convertScaleAbs(cv2.Sobel(g, cv2.CV_16S, 0, 1))
    return _a_bgr(cv2.addWeighted(gx, 0.5, gy, 0.5, 0))


def _umbral(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _a_bgr(t)


_SEPIA = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]], dtype=np.float32)


def _sepia(img):
    return cv2.transform(img, _SEPIA)


def _caricatura(img):
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    bordes = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 5)
    color = cv2.bilateralFilter(img, 7, 60, 60)
    return cv2.bitwise_and(color, color, mask=bordes)


def _pixelar(img, n=16):
    h, w = img.shape[:2]
    p = cv2.resize(img, (max(1, w // n), max(1, h // n)),
                   interpolation=cv2.INTER_LINEAR)
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)


def _termico(img):
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                             cv2.COLORMAP_JET)


def _contraste(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def filtro_kernel(k):
    return lambda img: cv2.filter2D(img, -1, k)


FILTROS = {
    "gris":       (_gris, "Escala de grises"),
    "desenfoque": (lambda i: cv2.GaussianBlur(i, (15, 15), 0), "Desenfoque gaussiano 15x15"),
    "bordes":     (_bordes, "Bordes con Canny"),
    "sobel":      (_sobel, "Gradiente Sobel (bordes suaves)"),
    "umbral":     (_umbral, "Blanco y negro automatico (Otsu)"),
    "sepia":      (_sepia, "Tono sepia"),
    "invertir":   (cv2.bitwise_not, "Negativo"),
    "caricatura": (_caricatura, "Efecto caricatura (mas lento)"),
    "pixelar":    (_pixelar, "Pixelado"),
    "termico":    (_termico, "Mapa de color tipo camara termica"),
    "contraste":  (_contraste, "Mejora de contraste (CLAHE)"),
    "relieve":    (filtro_kernel(np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)),
                   "Relieve (emboss)"),
    "nitidez":    (filtro_kernel(np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)),
                   "Enfocar"),
    "espejo":     (lambda i: cv2.flip(i, 1), "Voltear horizontal"),
}


def parsear_kernel(texto):
    """'0,-1,0;-1,5,-1;0,-1,0'  o  '0 -1 0 -1 5 -1 0 -1 0'  (+ 'norm' opcional)"""
    normalizar = "norm" in texto.lower()
    t = texto.lower().replace("norm", "")
    filas = [f for f in t.split(";") if f.strip()]
    if not filas:
        raise ValueError("Escribe la matriz, ej: kernel 0,-1,0;-1,5,-1;0,-1,0")
    if len(filas) == 1:
        vals = [float(v) for v in filas[0].replace(",", " ").split()]
        n = int(round(len(vals) ** 0.5))
        if n * n != len(vals):
            raise ValueError("Necesito una matriz cuadrada (9, 25... valores) o filas separadas por ';'")
        k = np.array(vals, np.float32).reshape(n, n)
    else:
        k = np.array([[float(v) for v in f.replace(",", " ").split()] for f in filas],
                     np.float32)
    if k.ndim != 2:
        raise ValueError("Todas las filas deben tener el mismo numero de valores")
    if normalizar and abs(k.sum()) > 1e-6:
        k /= k.sum()
    return k


# ---------------------------------------------------------------------------
# Detectores / modos de vision
# ---------------------------------------------------------------------------
def ruta_haar():
    nombre = "haarcascade_frontalface_default.xml"
    candidatos = []
    if hasattr(cv2, "data"):
        candidatos.append(os.path.join(cv2.data.haarcascades, nombre))
    candidatos += [f"/usr/share/opencv4/haarcascades/{nombre}",
                   f"/usr/share/opencv/haarcascades/{nombre}",
                   f"/usr/local/share/opencv4/haarcascades/{nombre}"]
    for c in candidatos:
        if os.path.exists(c):
            return c
    raise RuntimeError("No encontre el modelo de caras. Instala: sudo apt install opencv-data")


class DetCaras:
    def __init__(self, app):
        self.cc = cv2.CascadeClassifier(ruta_haar())

    def procesar(self, frame, salida):
        esc = 0.5
        g = cv2.equalizeHist(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        None, fx=esc, fy=esc))
        caras = self.cc.detectMultiScale(g, 1.15, 5, minSize=(30, 30))
        for (x, y, w, h) in caras:
            x, y, w, h = [int(v / esc) for v in (x, y, w, h)]
            cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return salida, f"{len(caras)} cara(s)"


class DetMovimiento:
    def __init__(self, app):
        self.app = app
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32,
                                                     detectShadows=False)
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ultima_foto = 0.0

    def procesar(self, frame, salida):
        h, w = frame.shape[:2]
        peq = cv2.resize(frame, (320, max(1, int(320 * h / w))))
        esc = w / 320
        m = self.bg.apply(cv2.GaussianBlur(peq, (5, 5), 0))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k)
        m = cv2.dilate(m, self.k, iterations=2)
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # compatible con OpenCV 3 y 4
        area_min = 0.005 * m.shape[0] * m.shape[1]
        n = 0
        for c in cnts:
            if cv2.contourArea(c) < area_min:
                continue
            n += 1
            x, y, bw, bh = [int(v * esc) for v in cv2.boundingRect(c)]
            cv2.rectangle(salida, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        if n:
            cv2.putText(salida, "MOVIMIENTO", (10, salida.shape[0] - 15),
                        FUENTE, 0.8, (0, 0, 255), 2)
            if self.app.alarma and time.time() - self.ultima_foto > 5:
                self.ultima_foto = time.time()
                ruta = self.app.guardar_foto(salida, "alarma")
                decir(f"\n[alarma] movimiento detectado -> {ruta}", "yellow")
        return salida, f"{n} zona(s) en movimiento"


class DetQR:
    def __init__(self, app):
        if not hasattr(cv2, "QRCodeDetector"):
            raise RuntimeError("Tu OpenCV es muy viejo para leer QR (necesitas 4.x)")
        self.qr = cv2.QRCodeDetector()
        self.ultimo = ""

    def procesar(self, frame, salida):
        texto, pts, _ = self.qr.detectAndDecode(frame)
        if pts is not None and len(pts):
            p = pts.reshape(-1, 2).astype(int)
            cv2.polylines(salida, [p], True, (255, 0, 255), 3)
            if texto:
                cv2.putText(salida, texto[:40], (int(p[0][0]), max(20, int(p[0][1]) - 10)),
                            FUENTE, 0.6, (255, 0, 255), 2)
                if texto != self.ultimo:
                    self.ultimo = texto
                    decir(f"\n[qr] {texto}", "magenta")
        return salida, (f"QR: {texto[:30]}" if texto else "buscando QR...")


def cargar_yolo(nombre, tarea):
    if not tiene("ultralytics"):
        raise RuntimeError("Falta ultralytics. Instala: pip install ultralytics ncnn")
    from ultralytics import YOLO
    ncnn = f"{nombre}_ncnn_model"
    if os.path.isdir(ncnn):
        decir(f"Usando modelo optimizado NCNN: {ncnn}", "green")
        return YOLO(ncnn, task=tarea)
    decir(f"Cargando {nombre}.pt (la primera vez se descarga). "
          "Tip: el comando 'optimizar' lo hace 2-3x mas rapido.", "yellow")
    return YOLO(f"{nombre}.pt", task=tarea)


class DetObjetos:
    def __init__(self, app):
        self.app = app
        self.modelo = cargar_yolo(app.args.modelo, "detect")

    def procesar(self, frame, salida):
        r = self.modelo.predict(frame, imgsz=self.app.imgsz, conf=self.app.conf,
                                classes=self.app.clases, verbose=False)[0]
        salida = r.plot(img=salida, line_width=2)
        cuenta = Counter(r.names[int(c)] for c in r.boxes.cls) if r.boxes is not None else Counter()
        return salida, ", ".join(f"{v} {k}" for k, v in cuenta.most_common(4)) or "sin objetos"


MP_MODELO = "pose_landmarker_lite.task"
MP_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
          "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
# Conexiones del esqueleto de 33 puntos de MediaPipe
MP_CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                 (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                 (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                 (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
                 (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]


class DetPose:
    def __init__(self, app):
        self.app = app
        self.motor = None
        pref = app.args.pose
        if pref in ("auto", "mediapipe") and tiene("mediapipe"):
            try:
                self._iniciar_mediapipe()
            except Exception as e:
                decir(f"MediaPipe no funciono ({e}); intento con YOLO-pose.", "yellow")
        if self.motor is None:
            if pref == "mediapipe":
                raise RuntimeError("MediaPipe no esta disponible: pip install mediapipe")
            if not tiene("ultralytics"):
                raise RuntimeError("Necesito MediaPipe funcionando (pip install mediapipe) "
                                   "o YOLO (pip install ultralytics ncnn)")
            self.modelo = cargar_yolo(app.args.modelo + "-pose", "pose")
            self.motor = "yolo"
        decir(f"Esqueleto con motor: {self.motor}", "green")

    def _iniciar_mediapipe(self):
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions"):          # MediaPipe antiguo (<= 0.10.2x)
            self.pose = mp.solutions.pose.Pose(model_complexity=0,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
            self.api = "solutions"
        else:                                 # MediaPipe nuevo: API "Tasks"
            from mediapipe.tasks.python import BaseOptions, vision
            descargar_modelo(MP_MODELO, MP_URL)
            opciones = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MP_MODELO),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2, min_pose_detection_confidence=0.5)
            self.pose = vision.PoseLandmarker.create_from_options(opciones)
            self.api = "tasks"
            self.t_ms = 0
        self.motor = "mediapipe"

    @staticmethod
    def _dibujar_esqueleto(img, puntos):
        h, w = img.shape[:2]
        pts = [(int(p.x * w), int(p.y * h), getattr(p, "visibility", 1.0) or 1.0) for p in puntos]
        for a, b in MP_CONEXIONES:
            if a < len(pts) and b < len(pts) and pts[a][2] > 0.5 and pts[b][2] > 0.5:
                cv2.line(img, pts[a][:2], pts[b][:2], (0, 255, 255), 2, cv2.LINE_AA)
        for x, y, v in pts:
            if v > 0.5:
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

    def procesar(self, frame, salida):
        if self.motor == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.api == "solutions":
                res = self.pose.process(rgb)
                personas = [res.pose_landmarks.landmark] if res.pose_landmarks else []
            else:
                self.t_ms = max(self.t_ms + 1, int(time.time() * 1000))
                img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
                personas = self.pose.detect_for_video(img, self.t_ms).pose_landmarks
            for puntos in personas:
                self._dibujar_esqueleto(salida, puntos)
            return salida, f"{len(personas)} persona(s)" if personas else "sin persona"
        r = self.modelo.predict(frame, imgsz=self.app.imgsz, conf=self.app.conf,
                                verbose=False)[0]
        n = 0 if r.keypoints is None else len(r.keypoints)
        return r.plot(img=salida, boxes=False), f"{n} persona(s)"


def descargar_modelo(archivo, url):
    if os.path.exists(archivo):
        return
    decir(f"Descargando {archivo} (solo la primera vez)...", "cyan")
    import urllib.request
    urllib.request.urlretrieve(url, archivo + ".tmp")
    os.replace(archivo + ".tmp", archivo)


# ---- Manos y gestos (MediaPipe) -------------------------------------------
GESTOS_MODELO = "gesture_recognizer.task"
GESTOS_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
              "gesture_recognizer/float16/latest/gesture_recognizer.task")
GESTOS_ES = {"Closed_Fist": "Puno cerrado", "Open_Palm": "Mano abierta",
             "Pointing_Up": "Senalando arriba", "Thumb_Down": "Pulgar abajo",
             "Thumb_Up": "Pulgar arriba", "Victory": "Amor y paz",
             "ILoveYou": "Te quiero (rock)", "None": ""}
MANO_CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                   (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def contar_dedos(p):
    """p: 21 puntos normalizados de una mano. Devuelve cuantos dedos estan levantados."""
    d = lambda a, b: ((p[a].x - p[b].x) ** 2 + (p[a].y - p[b].y) ** 2) ** 0.5
    n = int(d(4, 17) > d(3, 17) * 1.1 and d(4, 5) > d(3, 5))          # pulgar
    for punta, articulacion in ((8, 6), (12, 10), (16, 14), (20, 18)):
        n += int(d(punta, 0) > d(articulacion, 0) * 1.1)                  # resto
    return n


class DetManos:
    def __init__(self, app):
        if not tiene("mediapipe"):
            raise RuntimeError("Falta MediaPipe: pip install mediapipe")
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions"):          # MediaPipe antiguo: solo puntos + dedos
            self.det = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                                min_detection_confidence=0.6)
            self.api = "solutions"
        else:                                 # MediaPipe nuevo: reconoce gestos
            from mediapipe.tasks.python import BaseOptions, vision
            descargar_modelo(GESTOS_MODELO, GESTOS_URL)
            self.det = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_path=GESTOS_MODELO),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2))
            self.api = "tasks"
        self.t_ms = 0
        self.ultimo, self.candidato, self.repeticiones = "", "", 0

    def procesar(self, frame, salida):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.api == "solutions":
            res = self.det.process(rgb)
            manos = [m.landmark for m in (res.multi_hand_landmarks or [])]
            gestos = [""] * len(manos)
        else:
            self.t_ms = max(self.t_ms + 1, int(time.time() * 1000))
            res = self.det.recognize_for_video(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.t_ms)
            manos = res.hand_landmarks
            gestos = [GESTOS_ES.get(g[0].category_name, g[0].category_name) if g else ""
                      for g in res.gestures]
        h, w = salida.shape[:2]
        textos = []
        for puntos, gesto in zip(manos, gestos):
            pts = [(int(q.x * w), int(q.y * h)) for q in puntos]
            for a, b in MANO_CONEXIONES:
                cv2.line(salida, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(salida, (x, y), 3, (0, 0, 255), -1)
            dedos = contar_dedos(puntos)
            etiqueta = f"{dedos} dedo(s)" + (f" - {gesto}" if gesto else "")
            x0, y0 = min(x for x, _ in pts), min(y for _, y in pts)
            cv2.putText(salida, etiqueta, (x0, max(20, y0 - 10)), FUENTE, 0.6, (0, 0, 0), 3)
            cv2.putText(salida, etiqueta, (x0, max(20, y0 - 10)), FUENTE, 0.6, (0, 255, 0), 1)
            textos.append(etiqueta)
        resumen = " | ".join(textos)
        # avisa en la terminal cuando un gesto se mantiene estable ~6 cuadros
        if resumen == self.candidato:
            self.repeticiones += 1
        else:
            self.candidato, self.repeticiones = resumen, 0
        if self.repeticiones == 6 and resumen and resumen != self.ultimo:
            self.ultimo = resumen
            decir(f"\n[manos] {resumen}", "green")
        return salida, resumen or "sin manos"


# ---- Seguir un objeto por su color -----------------------------------------
COLORES = {  # rangos HSV de OpenCV (H 0-179)
    "rojo":     [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "naranja":  [((9, 120, 90), (22, 255, 255))],
    "amarillo": [((23, 100, 100), (35, 255, 255))],
    "verde":    [((36, 70, 50), (85, 255, 255))],
    "azul":     [((90, 100, 50), (130, 255, 255))],
    "morado":   [((131, 60, 50), (160, 255, 255))],
}


def rango_desde_pixel(hsv, x, y):
    """Crea un rango HSV a partir de un parche de 11x11 alrededor de (x, y)."""
    h, w = hsv.shape[:2]
    parche = hsv[max(0, y - 5):min(h, y + 6), max(0, x - 5):min(w, x + 6)].reshape(-1, 3)
    hh, ss, vv = [int(v) for v in np.median(parche, axis=0)]
    s_min, v_min = max(40, ss - 70), max(40, vv - 80)
    lo, hi = hh - 10, hh + 10
    if lo < 0:      # el rojo da la vuelta en el circulo de color
        return [((0, s_min, v_min), (hi, 255, 255)), ((180 + lo, s_min, v_min), (179, 255, 255))]
    if hi > 179:
        return [((lo, s_min, v_min), (179, 255, 255)), ((0, s_min, v_min), (hi - 180, 255, 255))]
    return [((lo, s_min, v_min), (hi, 255, 255))]


class DetColor:
    def __init__(self, app):
        self.app = app
        self.rastro = []
        if app.rango_color is None:
            app.rango_color, app.nombre_color = COLORES["rojo"], "rojo"
        decir("Siguiendo color: " + app.nombre_color +
              ". Cambialo con 'color azul' o haz clic sobre el objeto en la ventana.", "cyan")

    def procesar(self, frame, salida):
        if self.app.clic is not None:                 # clic en la ventana = elegir color
            x, y = self.app.clic
            self.app.clic = None
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.app.rango_color = rango_desde_pixel(hsv, x, y)
            self.app.nombre_color = f"clic ({x},{y})"
            self.rastro.clear()
            decir(f"\n[color] nuevo color tomado en ({x},{y})", "cyan")
        h, w = frame.shape[:2]
        esc = 320 / w
        peq = cv2.GaussianBlur(cv2.resize(frame, (320, max(1, int(h * esc)))), (5, 5), 0)
        hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
        mascara = None
        for lo, hi in self.app.rango_color:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mascara = m if mascara is None else cv2.bitwise_or(mascara, m)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mascara = cv2.dilate(mascara, None, iterations=2)
        cnts = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # compatible con OpenCV 3 y 4
        cnts = [c for c in cnts if cv2.contourArea(c) > 80]
        if not cnts:
            self.rastro = self.rastro[1:]
            info = f"color {self.app.nombre_color}: no visible"
        else:
            c = max(cnts, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            cx, cy, r = int(cx / esc), int(cy / esc), int(r / esc)
            cv2.circle(salida, (cx, cy), r, (0, 255, 255), 2)
            cv2.circle(salida, (cx, cy), 5, (0, 0, 255), -1)
            self.rastro = (self.rastro + [(cx, cy)])[-40:]
            # posicion relativa al centro (util para mover un robot o servo)
            dx, dy = cx - w // 2, cy - h // 2
            lado = "izq" if dx < -w * 0.1 else "der" if dx > w * 0.1 else "centro"
            info = f"color {self.app.nombre_color}: ({cx},{cy}) {lado}"
        for i in range(1, len(self.rastro)):
            grosor = max(1, int(i / 6))
            cv2.line(salida, self.rastro[i - 1], self.rastro[i], (255, 0, 255), grosor)
        if self.app.ver_mascara:
            mini = cv2.cvtColor(cv2.resize(mascara, (w // 4, h // 4)), cv2.COLOR_GRAY2BGR)
            salida[h - h // 4:h, w - w // 4:w] = mini
        return salida, info


MODOS = {
    "normal":     (None, "Solo video + filtros"),
    "caras":      (DetCaras, "Deteccion de caras (Haar, rapido, solo OpenCV)"),
    "movimiento": (DetMovimiento, "Detecta movimiento; con 'alarma on' guarda fotos"),
    "qr":         (DetQR, "Lee codigos QR"),
    "pose":       (DetPose, "Esqueleto de personas (MediaPipe o YOLO-pose)"),
    "objetos":    (DetObjetos, "Deteccion de objetos con YOLO (80 clases COCO)"),
    "manos":      (DetManos, "Manos: dedos levantados y gestos (MediaPipe)"),
    "color":      (DetColor, "Sigue un objeto por su color (clic en la ventana para elegirlo)"),
}


# ---------------------------------------------------------------------------
# Camara (hilo propio: siempre guarda el cuadro mas reciente)
# ---------------------------------------------------------------------------
class Camara:
    def __init__(self, fuente, ancho, alto):
        self.fuente = str(fuente)
        self.ancho, self.alto = ancho, alto
        self.frame, self.id = None, 0
        self.lock = threading.Lock()
        self.activo = True
        self.pedido = None
        self.t = 0
        self.cap = None
        self.es_archivo = False
        self.retardo = 0
        if self.fuente != "prueba":
            self._abrir()
        threading.Thread(target=self._loop, daemon=True).start()

    def _abrir(self):
        if self.fuente.isdigit():
            idx = int(self.fuente)
            cap = (cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux")
                   else cv2.VideoCapture(idx))
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.fuente)
            self.es_archivo = os.path.isfile(self.fuente)
        if not cap.isOpened():
            raise RuntimeError(f"No pude abrir la camara/fuente '{self.fuente}'. "
                               "Revisa con: v4l2-ctl --list-devices  (o prueba --camara 1)")
        self.cap = cap
        if self.es_archivo:
            self.retardo = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30)

    def resolucion(self):
        with self.lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
                return w, h
        return self.ancho, self.alto

    def pedir_resolucion(self, w, h):
        self.pedido = (w, h)

    def _patron(self):
        self.t += 1
        w, h = self.ancho, self.alto
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
        img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        img[:, :, 2] = 120
        cx = int((np.sin(self.t / 30) * 0.4 + 0.5) * w)
        cy = int((np.cos(self.t / 45) * 0.3 + 0.5) * h)
        cv2.circle(img, (cx, cy), max(10, w // 16), (255, 255, 255), -1)
        cv2.rectangle(img, (w // 6, h // 6), (w // 6 + w // 8, h // 6 + h // 8), (0, 0, 0), -1)
        cv2.putText(img, "PATRON DE PRUEBA", (20, h - 20), FUENTE, 0.8, (255, 255, 255), 2)
        return img

    def _loop(self):
        fallos = 0
        while self.activo:
            if self.pedido:
                w, h = self.pedido
                self.pedido = None
                self.ancho, self.alto = w, h
                if self.cap is not None and not self.es_archivo:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if self.cap is None:
                frame = self._patron()
                time.sleep(1 / 30)
            else:
                ok, frame = self.cap.read()
                if not ok:
                    if self.es_archivo:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    fallos += 1
                    if fallos == 50:
                        decir("\n[camara] no llegan imagenes. Se desconecto?", "red")
                    time.sleep(0.05)
                    continue
                fallos = 0
                if self.retardo:
                    time.sleep(self.retardo)
            with self.lock:
                self.frame = frame
                self.id += 1

    def leer(self):
        with self.lock:
            return self.frame, self.id

    def cerrar(self):
        self.activo = False
        time.sleep(0.1)
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------
class App:
    def __init__(self, args):
        self.args = args
        self.activo = True
        self.cadena = []            # [(nombre, funcion)]
        self.detector = None
        self.modo = "normal"
        self.conf = 0.45
        self.clases = None
        self.imgsz = args.imgsz
        self.alarma = False
        self.hud = True
        self.grabar = False
        self.writer = None
        self.ruta_video = ""
        self.fps = 0.0
        self.info = ""
        self.url = ""
        self.salida, self.salida_id = None, 0
        self.lock = threading.Lock()
        self.rango_color, self.nombre_color = None, ""
        self.ver_mascara = False
        self.clic = None
        self._raton = False
        self.ventana = (not args.sin_ventana and
                        bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
        os.makedirs(CARPETA, exist_ok=True)
        self.cam = Camara(args.camara, args.ancho, args.alto)
        self.hilo = threading.Thread(target=self._procesar, daemon=True)
        self.hilo.start()

    # ---- bucle de procesamiento -------------------------------------------
    def _procesar(self):
        ult = -1
        t_prev = time.time()
        while self.activo:
            frame, fid = self.cam.leer()
            if frame is None or fid == ult:
                time.sleep(0.003)
                continue
            ult = fid
            salida = frame
            for nombre, f in list(self.cadena):
                try:
                    salida = f(salida)
                except Exception as e:
                    decir(f"\n[error] el filtro '{nombre}' fallo: {e}. Lo quito.", "red")
                    self.cadena = [c for c in self.cadena if c[0] != nombre]
            if salida is frame:
                salida = frame.copy()

            info = ""
            det = self.detector
            if det is not None:
                try:
                    salida, info = det.procesar(frame, salida)
                except Exception as e:
                    decir(f"\n[error] el modo '{self.modo}' fallo: {e}. Vuelvo a normal.", "red")
                    self.detector, self.modo = None, "normal"
            self.info = info

            ahora = time.time()
            dt, t_prev = ahora - t_prev, ahora
            if dt > 0:
                self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt

            if self.hud:
                self._dibujar_hud(salida)
            if self.grabar:
                self._escribir(salida)
            elif self.writer is not None:
                self.writer.release()
                self.writer = None
                decir(f"\n[video] guardado en {self.ruta_video}", "green")

            with self.lock:
                self.salida = salida
                self.salida_id += 1
            if self.ventana:
                self._mostrar(salida)

    def _dibujar_hud(self, img):
        lineas = [f"{self.fps:4.1f} FPS  modo:{self.modo}  filtros:{self._nombres() or '-'}"]
        if self.info:
            lineas.append(self.info)
        if self.grabar:
            lineas.append("REC")
        for i, t in enumerate(lineas):
            y = 22 + i * 22
            cv2.putText(img, t, (8, y), FUENTE, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (8, y), FUENTE, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _mostrar(self, img):
        try:
            cv2.imshow("VisionPi", img)
            if not self._raton:
                cv2.setMouseCallback("VisionPi", self._al_clic)
                self._raton = True
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                decir("\nVentana cerrada. Presiona Enter para terminar.", "yellow")
                self.activo = False
        except cv2.error:
            self.ventana = False
            decir("\nNo se pudo abrir ventana; usa el visor web.", "yellow")

    def _al_clic(self, evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            self.clic = (x, y)
            if self.modo != "color":
                decir("\n[color] clic recibido; activa 'modo color' para seguir ese color.", "yellow")

    def _escribir(self, img):
        if self.writer is None:
            h, w = img.shape[:2]
            self.ruta_video = os.path.join(
                CARPETA, datetime.now().strftime("video_%Y%m%d_%H%M%S.avi"))
            fps = max(5, round(self.fps or 15))
            self.writer = cv2.VideoWriter(self.ruta_video, cv2.VideoWriter_fourcc(*"MJPG"),
                                          fps, (w, h))
        self.writer.write(img)

    # ---- utilidades -------------------------------------------------------
    def _nombres(self):
        return " > ".join(n for n, _ in self.cadena)

    def ultima(self):
        with self.lock:
            return self.salida, self.salida_id

    def guardar_foto(self, img=None, prefijo="foto"):
        if img is None:
            img, _ = self.ultima()
        if img is None:
            return None
        ruta = os.path.join(CARPETA, datetime.now().strftime(f"{prefijo}_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg")
        cv2.imwrite(ruta, img)
        return ruta

    def cerrar(self):
        self.activo = False
        self.grabar = False
        self.hilo.join(timeout=2)
        if self.writer is not None:
            self.writer.release()
        self.cam.cerrar()

    # ---- comandos ---------------------------------------------------------
    def comando(self, linea):
        linea = linea.strip()
        if not linea:
            return self.activo
        cmd, _, resto = linea.partition(" ")
        cmd, resto = cmd.lower(), resto.strip()
        partes = resto.split()

        if cmd in ("salir", "exit", "quit", "q"):
            return False
        elif cmd in ("ayuda", "help", "?", "h"):
            mostrar_ayuda()
        elif cmd == "filtros":
            tabla("Filtros", ["nombre", "descripcion"],
                  [(n, d) for n, (_, d) in FILTROS.items()])
            decir("Combinalos: 'filtro gris bordes'. Filtro propio: 'kernel 0,-1,0;-1,5,-1;0,-1,0'")
        elif cmd == "modos":
            tabla("Modos", ["modo", "descripcion"], [(n, d) for n, (_, d) in MODOS.items()])
        elif cmd == "filtro":
            self._cmd_filtro(partes)
        elif cmd == "kernel":
            try:
                k = parsear_kernel(resto)
            except ValueError as e:
                decir(str(e), "red")
                return True
            nombre = f"kernel{k.shape[0]}x{k.shape[1]}"
            self.cadena = self.cadena + [(nombre, filtro_kernel(k))]
            decir(f"Agregado {nombre}:\n{np.array2string(k, precision=3)}", "green")
            decir(f"Cadena: {self._nombres()}")
        elif cmd == "modo":
            self._cmd_modo(partes[0].lower() if partes else "")
        elif cmd == "conf":
            try:
                v = float(partes[0])
                assert 0 < v < 1
                self.conf = v
                decir(f"Confianza minima = {v}", "green")
            except (IndexError, ValueError, AssertionError):
                decir("Uso: conf 0.5   (entre 0 y 1)", "red")
        elif cmd == "clases":
            m = self._modelo_objetos()
            if m:
                decir(", ".join(v.replace(" ", "_") for v in m.names.values()))
        elif cmd == "solo":
            self._cmd_solo(partes)
        elif cmd == "optimizar":
            self._cmd_optimizar(partes[0].lower() if partes else "objetos")
        elif cmd == "res":
            try:
                w, h = [int(x) for x in partes[0].lower().split("x")]
            except (IndexError, ValueError):
                decir("Uso: res 640x480", "red")
                return True
            self.grabar = False
            self.cam.pedir_resolucion(w, h)
            time.sleep(0.5)
            rw, rh = self.cam.resolucion()
            decir(f"Resolucion pedida {w}x{h}; la camara entrega {rw}x{rh}", "green")
        elif cmd == "foto":
            ruta = self.guardar_foto()
            decir(f"Foto guardada: {ruta}" if ruta else "Aun no hay imagen", "green")
        elif cmd == "grabar":
            self.grabar = not self.grabar
            decir("Grabando... (escribe 'grabar' otra vez para detener)" if self.grabar
                  else "Deteniendo grabacion...", "green")
        elif cmd == "alarma":
            self.alarma = bool(partes) and partes[0].lower() in ("on", "si", "1")
            decir(f"Alarma {'activada' if self.alarma else 'desactivada'}", "green")
            if self.alarma and self.modo != "movimiento":
                decir("Nota: la alarma funciona en 'modo movimiento'.", "yellow")
        elif cmd == "hud":
            self.hud = not (partes and partes[0].lower() in ("off", "no", "0"))
            decir(f"HUD {'visible' if self.hud else 'oculto'}", "green")
        elif cmd == "color":
            self._cmd_color(partes)
        elif cmd == "mascara":
            self.ver_mascara = not self.ver_mascara
            decir(f"Mascara {'visible' if self.ver_mascara else 'oculta'}", "green")
        elif cmd == "estado":
            self._cmd_estado()
        else:
            decir(f"No conozco '{cmd}'. Escribe 'ayuda'.", "red")
        return self.activo

    def _cmd_filtro(self, partes):
        if not partes:
            decir(f"Filtros activos: {self._nombres() or 'ninguno'}")
            return
        if partes[0].lower() in ("ninguno", "quitar", "off", "0"):
            self.cadena = []
            decir("Filtros quitados", "green")
            return
        agregar = partes[0] == "+"
        nombres = [n.lower() for n in (partes[1:] if agregar else partes)]
        malos = [n for n in nombres if n not in FILTROS]
        if malos:
            decir(f"No conozco: {', '.join(malos)}. Escribe 'filtros' para ver la lista.", "red")
            return
        nueva = [(n, FILTROS[n][0]) for n in nombres]
        self.cadena = (self.cadena + nueva) if agregar else nueva
        decir(f"Filtros: {self._nombres()}", "green")

    def _cmd_modo(self, nombre):
        if nombre not in MODOS:
            decir(f"Modos: {', '.join(MODOS)}", "red")
            return
        clase = MODOS[nombre][0]
        if clase is None:
            self.detector = None
        else:
            decir(f"Activando modo {nombre}...", "cyan")
            try:
                det = clase(self)
            except Exception as e:
                decir(f"No se pudo activar '{nombre}': {e}", "red")
                return
            self.detector = det
        self.modo = nombre
        decir(f"Modo: {nombre}", "green")

    def _cmd_color(self, partes):
        if not partes:
            decir(f"Colores: {', '.join(COLORES)}, o 'color centro' para tomar el del centro. "
                  "Tambien puedes hacer clic en la ventana.")
            return
        nombre = partes[0].lower()
        if nombre == "centro":
            frame, _ = self.cam.leer()
            if frame is None:
                return
            h, w = frame.shape[:2]
            self.clic = (w // 2, h // 2)
            decir("Tomando el color del centro de la imagen.", "green")
        elif nombre in COLORES:
            self.rango_color, self.nombre_color = COLORES[nombre], nombre
            decir(f"Siguiendo color: {nombre}", "green")
        else:
            decir(f"No conozco '{nombre}'. Colores: {', '.join(COLORES)}, centro", "red")
            return
        if self.modo != "color":
            self._cmd_modo("color")

    def _modelo_objetos(self):
        if isinstance(self.detector, DetObjetos):
            return self.detector.modelo
        decir("Primero activa 'modo objetos'.", "red")
        return None

    def _cmd_solo(self, partes):
        if not partes or partes[0].lower() in ("todo", "todos", "todas"):
            self.clases = None
            decir("YOLO detectara todas las clases", "green")
            return
        m = self._modelo_objetos()
        if not m:
            return
        mapa = {v.replace(" ", "_").lower(): k for k, v in m.names.items()}
        ids, malos = [], []
        for p in partes:
            (ids.append(mapa[p.lower()]) if p.lower() in mapa else malos.append(p))
        if malos:
            decir(f"No existen: {', '.join(malos)}. Usa 'clases' para ver la lista.", "red")
        if ids:
            self.clases = ids
            decir(f"Detectando solo: {', '.join(m.names[i] for i in ids)}", "green")

    def _cmd_optimizar(self, cual):
        if not tiene("ultralytics"):
            decir("Falta ultralytics: pip install ultralytics ncnn", "red")
            return
        nombre = self.args.modelo + ("-pose" if cual == "pose" else "")
        decir(f"Exportando {nombre} a NCNN con imgsz={self.imgsz}. "
              "En la Pi tarda unos minutos...", "cyan")
        try:
            from ultralytics import YOLO
            YOLO(f"{nombre}.pt").export(format="ncnn", imgsz=self.imgsz)
        except Exception as e:
            decir(f"Fallo la exportacion: {e}", "red")
            return
        decir(f"Listo: {nombre}_ncnn_model. Se usara automaticamente.", "green")
        if (cual == "objetos" and self.modo == "objetos") or (cual == "pose" and self.modo == "pose"):
            self._cmd_modo(self.modo)

    def _cmd_estado(self):
        w, h = self.cam.resolucion()
        filas = [
            ("Fuente", self.cam.fuente),
            ("Resolucion", f"{w}x{h}"),
            ("FPS procesados", f"{self.fps:.1f}"),
            ("Modo", self.modo + (f"  ({self.info})" if self.info else "")),
            ("Filtros", self._nombres() or "ninguno"),
            ("YOLO conf / imgsz", f"{self.conf} / {self.imgsz}"),
            ("Clases", "todas" if self.clases is None else str(self.clases)),
            ("Grabando", "si" if self.grabar else "no"),
            ("Alarma", "on" if self.alarma else "off"),
            ("Color seguido", self.nombre_color or "-"),
            ("Ventana", "si" if self.ventana else "no"),
            ("Visor web", self.url or "apagado"),
        ]
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                filas.append(("Temperatura CPU", f"{int(f.read()) / 1000:.1f} C"))
        except (OSError, ValueError):
            pass
        tabla("Estado", ["", ""], filas)


def mostrar_ayuda():
    tabla("Comandos de VisionPi", ["comando", "que hace"], [
        ("ayuda", "Muestra esta ayuda"),
        ("modos / filtros", "Lista los modos de vision / los filtros"),
        ("modo <nombre>", "normal, caras, movimiento, qr, pose, objetos, manos, color"),
        ("color <nombre>|centro", "Sigue rojo, verde, azul...; o clic en la ventana"),
        ("mascara", "Muestra/oculta la mascara del color (para ajustar)"),
        ("filtro <a> [b ...]", "Filtros en cadena (ej: filtro gris bordes)"),
        ("filtro + <a>", "Agrega un filtro al final de la cadena"),
        ("filtro ninguno", "Quita todos los filtros"),
        ("kernel <matriz> [norm]", "Filtro propio (ej: kernel 0,-1,0;-1,5,-1;0,-1,0)"),
        ("conf <0-1>", "Confianza minima de YOLO (ej: conf 0.5)"),
        ("clases", "Clases que reconoce YOLO (en modo objetos)"),
        ("solo <clases> | solo todo", "YOLO solo detecta esas (ej: solo person cup)"),
        ("optimizar [objetos|pose]", "Convierte YOLO a NCNN (2-3x mas rapido en la Pi)"),
        ("res <ancho>x<alto>", "Cambia resolucion (ej: res 320x240)"),
        ("foto", "Guarda la imagen actual en ./capturas"),
        ("grabar", "Inicia/detiene grabacion de video (.avi)"),
        ("alarma on|off", "En modo movimiento guarda fotos automaticamente"),
        ("hud on|off", "Muestra/oculta el texto sobre el video"),
        ("estado", "FPS, modo, filtros, temperatura, direccion web"),
        ("salir", "Termina la app"),
    ])


# ---------------------------------------------------------------------------
# Visor web MJPEG (para ver el video por SSH desde el navegador)
# ---------------------------------------------------------------------------
PAGINA = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionPi</title>
<style>body{margin:0;background:#111;color:#ccc;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto;margin-top:8px}</style></head>
<body><h3>VisionPi</h3><img src="/stream"><p>Controla la app desde la terminal.</p></body></html>"""


def crear_servidor(app, puerto):
    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _enviar(self, tipo, datos):
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(datos)))
            self.end_headers()
            self.wfile.write(datos)

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    self._enviar("text/html; charset=utf-8", PAGINA)
                elif self.path.startswith("/foto"):
                    img, _ = app.ultima()
                    if img is None:
                        self.send_error(503)
                        return
                    self._enviar("image/jpeg", cv2.imencode(".jpg", img)[1].tobytes())
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    ult = -1
                    while app.activo:
                        img, i = app.ultima()
                        if img is None or i == ult:
                            time.sleep(0.01)
                            continue
                        ult = i
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n"
                                         + jpg.tobytes() + b"\r\n")
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("0.0.0.0", puerto), Manejador)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="VisionPi: vision por computadora en terminal")
    p.add_argument("--camara", default="0",
                   help="indice (0, 1...), 'prueba', archivo de video o URL")
    p.add_argument("--ancho", type=int, default=640)
    p.add_argument("--alto", type=int, default=480)
    p.add_argument("--puerto", type=int, default=8080, help="puerto del visor web")
    p.add_argument("--sin-web", action="store_true", help="no iniciar el visor web")
    p.add_argument("--sin-ventana", action="store_true", help="no abrir ventana de OpenCV")
    p.add_argument("--modelo", default="yolo11n", help="modelo YOLO base (yolo11n, yolov8n...)")
    p.add_argument("--imgsz", type=int, default=320, help="tamano de entrada YOLO (320 = rapido)")
    p.add_argument("--pose", default="auto", choices=["auto", "mediapipe", "yolo"],
                   help="motor para el esqueleto")
    args = p.parse_args()

    cv2.setNumThreads(4)
    decir("VisionPi - vision por computadora en terminal", "bold cyan")
    decir(f"OpenCV {cv2.__version__} | rich: {'si' if _console else 'no'} | "
          f"mediapipe: {'si' if tiene('mediapipe') else 'no'} | "
          f"ultralytics: {'si' if tiene('ultralytics') else 'no'}")

    try:
        app = App(args)
    except RuntimeError as e:
        decir(str(e), "red")
        sys.exit(1)

    if not args.sin_web:
        try:
            crear_servidor(app, args.puerto)
            app.url = f"http://{ip_local()}:{args.puerto}"
            decir(f"Visor web: {app.url}", "green")
        except OSError as e:
            decir(f"No pude abrir el puerto {args.puerto}: {e}", "yellow")
    if app.ventana:
        decir("Ventana de video activada (tecla q en la ventana = salir)", "green")
    decir("Escribe 'ayuda' para ver los comandos.\n")

    try:
        while app.activo:
            if not app.comando(input("visionpi> ")):
                break
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        app.cerrar()
        decir("Hasta luego!", "cyan")


if __name__ == "__main__":
    main()
