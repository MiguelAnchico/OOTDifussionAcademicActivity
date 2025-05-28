#!/usr/bin/env python3
"""
VTO Server - Servidor backend para Virtual Try-On
Recibe imagen de persona + número de ropa, retorna mejor resultado VTO

Ubicación: OOTDiffusion/backend/vto_server.py
Uso: python vto_server.py
"""

import os
import sys
import time
import subprocess
import json
import glob
from pathlib import Path
from datetime import datetime
from collections import deque
import statistics
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import del detector
from mobilenet_detector import MobileNetPersonDetector

class VTOMetrics:
    """Clase para rastrear métricas de rendimiento"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.processing_times = deque(maxlen=max_history)
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.category_times = {
            'upperbody': deque(maxlen=max_history),
            'lowerbody': deque(maxlen=max_history),
            'dress': deque(maxlen=max_history)
        }
    
    def add_processing_time(self, duration, category='upperbody', success=True):
        """Registra tiempo de procesamiento"""
        self.processing_times.append(duration)
        
        if category in self.category_times:
            self.category_times[category].append(duration)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_average_time(self):
        """Obtiene tiempo promedio total"""
        if not self.processing_times:
            return 0
        return statistics.mean(self.processing_times)
    
    def get_median_time(self):
        """Obtiene tiempo mediano total"""
        if not self.processing_times:
            return 0
        return statistics.median(self.processing_times)
    
    def get_category_average(self, category):
        """Obtiene tiempo promedio por categoría"""
        if category not in self.category_times or not self.category_times[category]:
            return 0
        return statistics.mean(self.category_times[category])
    
    def get_recent_performance(self, last_n=10):
        """Obtiene rendimiento de las últimas N ejecuciones"""
        if len(self.processing_times) < last_n:
            recent_times = list(self.processing_times)
        else:
            recent_times = list(self.processing_times)[-last_n:]
        
        if not recent_times:
            return {"count": 0, "average": 0, "min": 0, "max": 0}
        
        return {
            "count": len(recent_times),
            "average": statistics.mean(recent_times),
            "min": min(recent_times),
            "max": max(recent_times)
        }
    
    def get_uptime(self):
        """Obtiene tiempo de actividad del servidor"""
        return time.time() - self.start_time
    
    def get_success_rate(self):
        """Obtiene tasa de éxito"""
        total = self.success_count + self.error_count
        if total == 0:
            return 0
        return (self.success_count / total) * 100
    
    def get_metrics_summary(self):
        """Obtiene resumen completo de métricas"""
        uptime = self.get_uptime()
        total_requests = self.success_count + self.error_count
        
        return {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "total_requests": total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate_percent": self.get_success_rate(),
            "processing_times": {
                "total_samples": len(self.processing_times),
                "average_seconds": self.get_average_time(),
                "median_seconds": self.get_median_time(),
                "recent_10": self.get_recent_performance(10),
                "recent_5": self.get_recent_performance(5)
            },
            "category_performance": {
                "upperbody": {
                    "samples": len(self.category_times['upperbody']),
                    "average_seconds": self.get_category_average('upperbody')
                },
                "lowerbody": {
                    "samples": len(self.category_times['lowerbody']),
                    "average_seconds": self.get_category_average('lowerbody')
                },
                "dress": {
                    "samples": len(self.category_times['dress']),
                    "average_seconds": self.get_category_average('dress')
                }
            }
        }

class VTOServer:
    def __init__(self):
        self.temp_dir = "temp"
        self.clothe_dir = "clothe"
        self.results_dir = "../run/images_output"
        
        # Inicializar métricas
        self.metrics = VTOMetrics()
        
        # Crear directorios necesarios
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.clothe_dir, exist_ok=True)
        
        # Inicializar detector
        print("🤖 Inicializando MobileNet-SSD...")
        self.detector = MobileNetPersonDetector(models_dir="models")
        print("✅ VTO Server listo")
        
        # Limpiar archivos temporales antiguos
        self.cleanup_old_files()
    
    def cleanup_old_files(self):
        """Limpia archivos temporales antiguos"""
        try:
            # Limpiar temp
            temp_files = glob.glob(os.path.join(self.temp_dir, "*"))
            for file in temp_files:
                if os.path.isfile(file):
                    os.remove(file)
            
            # Limpiar results antiguos (más de 1 hora)
            current_time = time.time()
            result_files = glob.glob(os.path.join(self.results_dir, "*"))
            for file in result_files:
                if os.path.isfile(file) and current_time - os.path.getctime(file) > 3600:
                    os.remove(file)
                    
            print("🧹 Archivos temporales limpiados")
        except Exception as e:
            print(f"⚠️ Error limpiando archivos: {e}")
    
    def detect_garment_category(self, clothe_path):
        """Detecta automáticamente la categoría de la prenda"""
        filename = os.path.basename(clothe_path).lower()
        
        # Palabras clave para vestidos/full body
        dress_keywords = ['dress', 'vestido', 'gown', 'jumpsuit', 'overall', 'mono']
        
        # Palabras clave para parte inferior
        lower_keywords = ['pants', 'jean', 'skirt', 'short', 'trouser', 'pantalon', 'falda']
        
        # Palabras clave para parte superior  
        upper_keywords = ['shirt', 'blouse', 'top', 'jacket', 'sweater', 'hoodie', 'camisa', 'blusa', 'chaqueta']
        
        # Verificar vestidos primero
        for keyword in dress_keywords:
            if keyword in filename:
                return 2, 'dress', 'dc'  # category, name, model_type
        
        # Verificar parte inferior
        for keyword in lower_keywords:
            if keyword in filename:
                return 1, 'lowerbody', 'dc'  # category, name, model_type
        
        # Verificar parte superior
        for keyword in upper_keywords:
            if keyword in filename:
                return 0, 'upperbody', 'hd'  # category, name, model_type
        
        # Por defecto: parte superior (funciona con modelo hd)
        return 0, 'upperbody', 'hd'

    def get_clothe_list(self):
        """Obtiene lista de ropas disponibles con detección de categoría"""
        clothe_extensions = ['.jpg', '.jpeg', '.png']
        clothes = []
        
        for ext in clothe_extensions:
            clothes.extend(glob.glob(os.path.join(self.clothe_dir, f"*{ext}")))
            clothes.extend(glob.glob(os.path.join(self.clothe_dir, f"*{ext.upper()}")))
        
        # Ordenar y enumerar
        clothes.sort()
        clothe_list = []
        for i, clothe_path in enumerate(clothes, 1):
            clothe_name = os.path.basename(clothe_path)
            category, category_name, model_type = self.detect_garment_category(clothe_path)
            
            clothe_list.append({
                "id": i,
                "name": clothe_name,
                "path": clothe_path,
                "category": category,
                "category_name": category_name,
                "model_type": model_type
            })
        
        return clothe_list
    
    def get_clothe_by_id(self, clothe_id):
        """Obtiene ropa por ID con información de categoría"""
        clothes = self.get_clothe_list()
        
        for clothe in clothes:
            if clothe["id"] == clothe_id:
                return clothe
        
        return None
    
    async def process_person_image(self, image_file):
        """Procesa imagen de persona con MobileNet-SSD"""
        timestamp = int(time.time())
        
        # Guardar imagen subida
        input_path = os.path.join(self.temp_dir, f"input_person_{timestamp}.jpg")
        with open(input_path, "wb") as f:
            content = await image_file.read()
            f.write(content)
        
        # Procesar con MobileNet-SSD - REDUCIDO PARA AHORRAR MEMORIA GPU
        processed_path = os.path.join(self.temp_dir, f"person_{timestamp}.jpg")
        
        result = self.detector.process_person_image(
            input_path=input_path,
            output_path=processed_path,
            target_size=(512, 768)  # Reducido de (768, 1024) para ahorrar memoria GPU
        )
        
        # Limpiar imagen original
        os.remove(input_path)
        
        if result['success']:
            return processed_path, result
        else:
            raise Exception("Error procesando imagen de persona")
    
    def run_vto(self, person_path, clothe_info):
        """Ejecuta VTO y retorna mejor resultado"""
        # Cambiar al directorio run
        original_dir = os.getcwd()
        os.chdir("../run")
        
        try:
            # Convertir rutas a absolutas ANTES de cambiar directorio
            relative_person_path = os.path.abspath(os.path.join(original_dir, person_path))
            absolute_clothe_path = os.path.abspath(os.path.join(original_dir, clothe_info["path"]))
            
            # Comando VTO con rutas absolutas y categoría detectada
            cmd = [
                "python", "run_ootd.py",
                "--model_path", relative_person_path,
                "--cloth_path", absolute_clothe_path,
                "--model_type", clothe_info["model_type"],  # hd o dc según la prenda
                "--category", str(clothe_info["category"]),  # 0, 1, o 2
                "--scale", "1.5",  # Reducido de 2.0 para ahorrar memoria GPU
                "--sample", "2"    # Reducido de 4 para ahorrar memoria GPU
            ]
            
            print(f"🚀 Ejecutando VTO: {' '.join(cmd)}")
            print(f"👗 Categoría detectada: {clothe_info['category_name']} (modelo: {clothe_info['model_type']})")
            
            # Ejecutar comando
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ VTO completado")
                
                # Buscar resultados generados (archivos más recientes)
                result_files = glob.glob(f"images_output/out_{clothe_info['model_type']}_*.png")
                
                if result_files:
                    # Ordenar por tiempo de creación
                    result_files.sort(key=os.path.getctime, reverse=True)
                    
                    # CON 2 MUESTRAS, USAR EL MÁS RECIENTE
                    if len(result_files) >= 1:
                        best_result = result_files[0]  # Primer resultado (más reciente)
                        print(f"🎯 Resultado seleccionado: {best_result}")
                        return best_result
                    else:
                        raise Exception("No se generaron resultados")
                else:
                    raise Exception("No se encontraron archivos de resultado")
            else:
                raise Exception(f"Error en VTO: {result.stderr}")
                
        finally:
            # Volver al directorio original
            os.chdir(original_dir)
    
    async def process_vto_request(self, person_image: UploadFile, clothe_id: int):
        """Procesa solicitud completa de VTO"""
        start_time = time.time()
        success = False
        category_name = 'upperbody'  # default
        
        try:
            print(f"🚀 Iniciando VTO - Ropa ID: {clothe_id}")
            
            # Validar imagen
            if person_image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
                raise HTTPException(status_code=400, detail="Formato de imagen no soportado")
            
            # Obtener información completa de la ropa
            clothe_info = self.get_clothe_by_id(clothe_id)
            if not clothe_info:
                clothes = self.get_clothe_list()
                available_ids = [c["id"] for c in clothes]
                raise HTTPException(
                    status_code=404, 
                    detail=f"Ropa ID {clothe_id} no encontrada. IDs disponibles: {available_ids}"
                )
            
            category_name = clothe_info['category_name']
            print(f"👗 Ropa seleccionada: {clothe_info['name']} ({category_name})")
            
            # Procesar imagen de persona
            print("👤 Procesando imagen de persona...")
            processed_person_path, detection_info = await self.process_person_image(person_image)
            
            # Ejecutar VTO
            print("🎨 Generando virtual try-on...")
            best_result_path = self.run_vto(processed_person_path, clothe_info)
            
            # Limpiar imagen temporal de persona
            os.remove(processed_person_path)
            
            processing_time = time.time() - start_time
            success = True
            
            print(f"✅ VTO completado en {processing_time:.2f}s")
            print(f"📊 Tiempo promedio actual: {self.metrics.get_average_time():.2f}s")
            
            # Solo devolver la ruta/URL de la imagen
            result_filename = os.path.basename(best_result_path)
            
            return {
                "success": True,
                "image_url": f"/result/{result_filename}",
                "processing_time": processing_time,
                "clothe_id": clothe_id,
                "clothe_name": clothe_info["name"],
                "category": clothe_info["category_name"],
                "model_type": clothe_info["model_type"],
                "metrics": {
                    "average_time": self.metrics.get_average_time(),
                    "category_average": self.metrics.get_category_average(category_name),
                    "recent_performance": self.metrics.get_recent_performance(5)
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error en VTO: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
        finally:
            # Registrar métricas independientemente del resultado
            self.metrics.add_processing_time(
                time.time() - start_time, 
                category_name, 
                success
            )

# Crear instancia del servidor
vto_server = VTOServer()

# Crear app FastAPI
app = FastAPI(
    title="VTO Server API",
    description="Servidor de Virtual Try-On con MobileNet-SSD + OOTDiffusion",
    version="1.0.0"
)

# Habilitar CORS para requests desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint de salud del servidor"""
    metrics = vto_server.metrics.get_metrics_summary()
    return {
        "message": "VTO Server funcionando",
        "status": "healthy",
        "timestamp": time.time(),
        "uptime_hours": metrics["uptime_hours"],
        "total_requests": metrics["total_requests"],
        "average_processing_time": metrics["processing_times"]["average_seconds"]
    }

@app.get("/clothes")
async def get_clothes():
    """Obtiene lista de ropas disponibles"""
    try:
        clothes = vto_server.get_clothe_list()
        return {
            "success": True,
            "clothes": clothes,
            "total": len(clothes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vto")
async def virtual_try_on(
    person_image: UploadFile = File(..., description="Imagen de la persona"),
    clothe_id: int = Form(..., description="ID de la ropa a probar")
):
    """
    Endpoint principal de Virtual Try-On
    
    - **person_image**: Imagen de la persona (JPEG/PNG)
    - **clothe_id**: ID de la ropa (obtener IDs con /clothes)
    
    Retorna la mejor imagen generada con métricas de rendimiento
    """
    result = await vto_server.process_vto_request(person_image, clothe_id)
    
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/result/{filename}")
async def get_result_image(filename: str):
    """Descarga imagen resultado"""
    file_path = os.path.join("../run/images_output", filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/png",
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

@app.get("/metrics")
async def get_metrics():
    """Obtiene métricas detalladas del servidor"""
    return {
        "success": True,
        "metrics": vto_server.metrics.get_metrics_summary(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Check de salud detallado"""
    clothe_count = len(vto_server.get_clothe_list())
    metrics = vto_server.metrics.get_metrics_summary()
    
    return {
        "status": "healthy",
        "detector_ready": vto_server.detector.model_loaded,
        "clothe_count": clothe_count,
        "temp_dir": vto_server.temp_dir,
        "clothe_dir": vto_server.clothe_dir,
        "timestamp": time.time(),
        "performance": {
            "uptime_hours": metrics["uptime_hours"],
            "total_requests": metrics["total_requests"],
            "success_rate": metrics["success_rate_percent"],
            "average_time": metrics["processing_times"]["average_seconds"],
            "recent_average": metrics["processing_times"]["recent_5"]["average"]
        }
    }

if __name__ == "__main__":
    print("🚀 Iniciando VTO Server...")
    print("📁 Asegúrate de tener ropas en: backend/clothe/")
    print("🌐 Server corriendo en: http://0.0.0.0:8384")
    print("📖 Documentación API: http://0.0.0.0:8384/docs")
    print("📊 Métricas disponibles en: http://0.0.0.0:8384/metrics")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8384,
        reload=False
    )