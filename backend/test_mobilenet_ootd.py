#!/usr/bin/env python3
"""
Pipeline simple: MobileNet-SSD + OOTDiffusion
"""

import os
import sys
import argparse
import subprocess
import time

# Import del detector
from mobilenet_detector import MobileNetPersonDetector

class SimpleVTOPipeline:
    def __init__(self):
        self.temp_dir = "temp"
        self.results_dir = "../run/images_output"
        
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Inicializar detector
        self.detector = MobileNetPersonDetector(models_dir="models")
    
    def process_person(self, person_path):
        """Procesa imagen de persona con MobileNet-SSD"""
        print("👤 Procesando imagen de persona...")
        
        timestamp = int(time.time())
        processed_path = os.path.join(self.temp_dir, f"person_{timestamp}.jpg")
        
        result = self.detector.process_person_image(
            input_path=person_path,
            output_path=processed_path,
            target_size=(768, 1024)
        )
        
        if result['success']:
            print(f"✅ Persona procesada: {processed_path}")
            return processed_path
        else:
            raise Exception("Error procesando persona")
    
    def run_ootd(self, person_path, cloth_path):
        """Ejecuta OOTDiffusion con el comando que YA FUNCIONA"""
        print("👗 Ejecutando OOTDiffusion...")
        
        # DEBUG: Mostrar directorio actual
        print(f"📍 Directorio actual: {os.getcwd()}")
        print(f"📁 Contenido actual: {os.listdir('.')}")
        print(f"📁 Directorio padre: {os.listdir('..')}")
        
        # Verificar si existe ../run antes de cambiar
        if os.path.exists("../run"):
            print("✅ Directorio ../run existe")
        else:
            print("❌ Directorio ../run NO existe")
            print(f"📁 Contenido de directorio padre: {os.listdir('..')}")
        
        # Cambiar al directorio run
        print("🔄 Cambiando a ../run...")
        os.chdir("../run")
        
        # Comando que YA FUNCIONA
        cmd = [
            "python", "run_ootd.py",
            "--model_path", person_path,
            "--cloth_path", cloth_path,
            "--scale", "2.0",
            "--sample", "4"
        ]
        
        print(f"🚀 Ejecutando: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ OOTDiffusion completado")
                return True
            else:
                print(f"❌ Error OOTDiffusion: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando comando: {e}")
            return False
        finally:
            # Volver al directorio backend
            os.chdir("../backend")
    
    def run_pipeline(self, person_path, cloth_path):
        """Pipeline completo"""
        start_time = time.time()
        
        print("🚀 INICIANDO PIPELINE VTO")
        print("="*50)
        
        try:
            # Paso 1: Procesar persona
            processed_person = self.process_person(person_path)
            
            # Ajustar ruta para run_ootd.py
            relative_person_path = f"../backend/{processed_person}"
            
            # Paso 2: Ejecutar OOTDiffusion
            success = self.run_ootd(relative_person_path, cloth_path)
            
            # Limpiar archivo temporal
            if os.path.exists(processed_person):
                os.remove(processed_person)
            
            total_time = time.time() - start_time
            
            print("="*50)
            if success:
                print(f"🎉 PIPELINE COMPLETADO en {total_time:.2f}s")
                print(f"📁 Resultados en: {self.results_dir}")
            else:
                print("❌ PIPELINE FALLÓ")
            print("="*50)
            
            return success
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Pipeline MobileNet-SSD + OOTDiffusion')
    parser.add_argument('--person', required=True, help='Imagen de persona')
    parser.add_argument('--cloth', required=True, help='Imagen de ropa')
    
    args = parser.parse_args()
    
    # Verificar archivos
    if not os.path.exists(args.person):
        print(f"❌ No existe: {args.person}")
        return
    
    if not os.path.exists(args.cloth):
        print(f"❌ No existe: {args.cloth}")
        return
    
    # Ejecutar pipeline
    pipeline = SimpleVTOPipeline()
    success = pipeline.run_pipeline(args.person, args.cloth)
    
    if success:
        print("✅ Éxito - Revisa ../run/images_output/ para los resultados")
    else:
        print("❌ Falló")

if __name__ == "__main__":
    main()