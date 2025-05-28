#!/usr/bin/env python3
"""
MobileNet-SSD Person Detector for OOTDiffusion Backend
Detecta personas y prepara imágenes para OOTDiffusion VTO

Ubicación: OOTDiffusion/backend/mobilenet_detector.py
"""

import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path
import time

class MobileNetPersonDetector:
    def __init__(self, models_dir="models"):
        """
        Inicializa el detector MobileNet-SSD
        
        Args:
            models_dir: Directorio donde guardar/cargar los modelos
        """
        self.models_dir = models_dir
        self.net = None
        self.model_loaded = False
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        self.person_class_id = 15  # ID de 'person' en PASCAL VOC
        
        # Crear directorio de modelos
        os.makedirs(self.models_dir, exist_ok=True)
        
        print("🤖 Inicializando MobileNet-SSD Person Detector...")
        self.setup_model()
    
    def setup_model(self):
        """
        Descarga y configura el modelo MobileNet-SSD
        """
        # Rutas de los archivos del modelo
        prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
        
        prototxt_path = os.path.join(self.models_dir, "MobileNetSSD_deploy.prototxt")
        model_path = os.path.join(self.models_dir, "MobileNetSSD_deploy.caffemodel")
        
        try:
            # Descargar prototxt si no existe
            if not os.path.exists(prototxt_path):
                print("📥 Descargando configuración MobileNet-SSD...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
                print("✅ Configuración descargada")
            
            # Descargar modelo si no existe
            if not os.path.exists(model_path):
                print("📥 Descargando modelo MobileNet-SSD (~23MB)...")
                print("⏳ Esto puede tardar unos minutos la primera vez...")
                
                # Descarga con progreso
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, (downloaded * 100) // total_size)
                        print(f"\r📊 Descarga: {percent}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="", flush=True)
                
                urllib.request.urlretrieve(model_url, model_path, reporthook=show_progress)
                print("\n✅ Modelo descargado")
            
            # Cargar modelo
            print("🔄 Cargando modelo MobileNet-SSD...")
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            # Configurar backend (GPU si está disponible)
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("🚀 GPU detectada, usando CUDA para aceleración")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                print("💻 Usando CPU para inferencia")
            
            self.model_loaded = True
            print("✅ MobileNet-SSD cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando MobileNet-SSD: {e}")
            print("⚠️  El detector funcionará en modo fallback (centrado simple)")
            self.model_loaded = False
    
    def detect_persons(self, image):
        """
        Detecta personas en la imagen
        
        Args:
            image: Imagen OpenCV (BGR)
            
        Returns:
            Lista de detecciones [(x1, y1, x2, y2, confidence), ...]
        """
        if not self.model_loaded:
            return []
        
        (h, w) = image.shape[:2]
        
        # Crear blob para MobileNet-SSD
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=0.007843,  # 1/127.5
            size=(300, 300),       # Tamaño de entrada
            mean=127.5,           # Media para normalización
            swapRB=True           # OpenCV=BGR, modelo=RGB
        )
        
        # Ejecutar detección
        self.net.setInput(blob)
        detections = self.net.forward()
        
        persons = []
        
        # Procesar detecciones
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            # Solo personas con confianza suficiente
            if class_id == self.person_class_id and confidence > self.confidence_threshold:
                # Obtener coordenadas del bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Validar coordenadas
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    persons.append((x1, y1, x2, y2, confidence))
        
        return persons
    
    def apply_nms(self, persons):
        """
        Aplica Non-Maximum Suppression para eliminar detecciones duplicadas
        """
        if not persons:
            return []
        
        boxes = []
        confidences = []
        
        for (x1, y1, x2, y2, confidence) in persons:
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
        
        # Aplicar NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        # Filtrar detecciones
        filtered_persons = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]
                confidence = confidences[i]
                filtered_persons.append((x, y, x + w, y + h, confidence))
        
        return filtered_persons
    
    def get_best_person_crop(self, image, padding_factor=0.2):
        """
        Detecta y obtiene el mejor recorte de persona
        
        Args:
            image: Imagen OpenCV
            padding_factor: Factor de padding adicional
            
        Returns:
            tuple: (cropped_image, detection_info) o (None, None) si no hay detección
        """
        persons = self.detect_persons(image)
        
        if not persons:
            return None, None
        
        # Aplicar NMS
        filtered_persons = self.apply_nms(persons)
        
        if not filtered_persons:
            return None, None
        
        # Seleccionar la persona más grande
        best_person = max(filtered_persons, key=lambda p: (p[2]-p[0]) * (p[3]-p[1]))
        x1, y1, x2, y2, confidence = best_person
        
        # Calcular padding
        w = x2 - x1
        h = y2 - y1
        padding_w = int(w * padding_factor)
        padding_h = int(h * padding_factor)
        
        # Aplicar padding
        x1_pad = max(0, x1 - padding_w)
        y1_pad = max(0, y1 - padding_h)
        x2_pad = min(image.shape[1], x2 + padding_w)
        y2_pad = min(image.shape[0], y2 + padding_h)
        
        # Recortar imagen
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        detection_info = {
            'confidence': confidence,
            'original_bbox': (x1, y1, x2, y2),
            'padded_bbox': (x1_pad, y1_pad, x2_pad, y2_pad),
            'crop_size': cropped.shape
        }
        
        return cropped, detection_info
    
    def resize_and_center(self, image, target_size=(768, 1024)):
        """
        Redimensiona y centra imagen manteniendo proporción
        
        Args:
            image: Imagen OpenCV
            target_size: (width, height) objetivo
            
        Returns:
            Imagen redimensionada y centrada
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Calcular escala
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crear imagen final con fondo negro
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Centrar
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return result
    
    def process_person_image(self, input_path, output_path, target_size=(768, 1024), padding_factor=0.2):
        """
        Procesa imagen de persona para OOTDiffusion
        
        Args:
            input_path: Ruta de imagen de entrada
            output_path: Ruta donde guardar resultado
            target_size: Tamaño objetivo (width, height)
            padding_factor: Factor de padding
            
        Returns:
            dict: Información del procesamiento
        """
        start_time = time.time()
        
        # Cargar imagen
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {input_path}")
        
        print(f"📸 Procesando: {os.path.basename(input_path)}")
        print(f"📏 Tamaño original: {image.shape[1]}x{image.shape[0]}")
        
        # Intentar detectar y recortar persona
        if self.model_loaded:
            cropped, detection_info = self.get_best_person_crop(image, padding_factor)
            
            if cropped is not None:
                print(f"✅ Persona detectada (confianza: {detection_info['confidence']:.2f})")
                print(f"📦 Recorte: {detection_info['crop_size'][1]}x{detection_info['crop_size'][0]}")
                
                # Redimensionar y centrar
                final_image = self.resize_and_center(cropped, target_size)
                method_used = "detection"
            else:
                print("⚠️  No se detectó persona, usando imagen completa")
                final_image = self.resize_and_center(image, target_size)
                method_used = "fallback"
                detection_info = None
        else:
            print("⚠️  Detector no disponible, usando imagen completa")
            final_image = self.resize_and_center(image, target_size)
            method_used = "fallback"
            detection_info = None
        
        # Guardar resultado
        cv2.imwrite(output_path, final_image)
        
        processing_time = time.time() - start_time
        
        print(f"💾 Imagen guardada: {output_path}")
        print(f"📏 Tamaño final: {target_size[0]}x{target_size[1]}")
        print(f"⏱️  Tiempo de procesamiento: {processing_time:.2f}s")
        
        # Información de retorno
        result_info = {
            'success': True,
            'method_used': method_used,
            'processing_time': processing_time,
            'input_size': (image.shape[1], image.shape[0]),
            'output_size': target_size,
            'output_path': output_path,
            'detection_info': detection_info
        }
        
        return result_info

# Función de conveniencia para uso directo
def process_person_for_vto(input_path, output_path, target_size=(768, 1024)):
    """
    Función de conveniencia para procesar una imagen de persona
    
    Args:
        input_path: Ruta de imagen de entrada
        output_path: Ruta de salida
        target_size: Tamaño objetivo (width, height)
        
    Returns:
        dict: Información del procesamiento
    """
    detector = MobileNetPersonDetector()
    return detector.process_person_image(input_path, output_path, target_size)

if __name__ == "__main__":
    # Prueba del detector
    import argparse
    
    parser = argparse.ArgumentParser(description='Procesar imagen de persona para VTO')
    parser.add_argument('--input', required=True, help='Imagen de entrada')
    parser.add_argument('--output', required=True, help='Imagen de salida')
    parser.add_argument('--width', type=int, default=768, help='Ancho objetivo')
    parser.add_argument('--height', type=int, default=1024, help='Alto objetivo')
    
    args = parser.parse_args()
    
    try:
        result = process_person_for_vto(
            input_path=args.input,
            output_path=args.output,
            target_size=(args.width, args.height)
        )
        
        print("\n" + "="*50)
        print("🎉 PROCESAMIENTO COMPLETADO")
        print(f"✅ Método usado: {result['method_used']}")
        print(f"⏱️  Tiempo total: {result['processing_time']:.2f}s")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
