# OOTDiffusion Implementation Guide

Este repositorio contiene la implementación y guía completa para instalar **OOTDiffusion** para virtual try-on, basado en el trabajo original de [levihsu/OOTDiffusion](https://github.com/levihsu/OOTDiffusion).

## 📋 Tabla de Contenidos

- [Hardware Utilizado](#hardware-utilizado-en-las-pruebas)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Instalación de OOTDiffusion](#instalación-de-ootdiffusion)
- [Descarga de Modelos](#descarga-de-modelos)
- [Configuración del Entorno](#configuración-del-entorno)
- [Uso Básico](#uso-básico)
- [Backend API Server](#backend-api-server)
- [Solución de Problemas](#solución-de-problemas)
- [Transferencia de Archivos](#transferencia-de-archivos)
- [Otros Modelos GAN](#otros-modelos-gan)
- [Créditos y Licencia](#créditos-y-licencia)

## 🖥️ Hardware Utilizado en las Pruebas

- **Entorno**: Cloud GPU con 80GB de almacenamiento
- **OS**: Linux
- **CUDA**: Disponible y compatible
- **Python**: 3.10
- **Tarjeta Gráfica (GPU)**:  RTX 3090 recomendado
- **GPU Memory**: Mínimo 12GB VRAM recomendado

## 🏗️ Arquitectura del Modelo

OOTDiffusion utiliza una arquitectura de difusión avanzada que combina múltiples componentes para lograr un virtual try-on realista:

![Esquema OOTDiffusion](workflow_ootd.png)

### Componentes principales:
- **VAE Encoder/Decoder**: Codificación y decodificación de imágenes
- **CLIP Image/Text Encoder**: Comprensión de imágenes y texto
- **Outfitting UNet**: Red neuronal especializada en fusión de prendas
- **Denoising UNet**: Proceso de eliminación de ruido en múltiples pasos
- **Mask Generator**: Generación de máscaras para áreas específicas

El proceso combina la imagen de la persona, la prenda objetivo y opcionalmente etiquetas de categoría para generar resultados precisos.

## 🚀 Instalación de OOTDiffusion

### Paso 1: Configuración inicial del entorno

```bash
# Verificar que tenemos CUDA disponible
nvidia-smi

# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y git git-lfs build-essential cmake pkg-config libgl1-mesa-glx
```

### Paso 2: Clonar y configurar OOTDiffusion

```bash
# Clonar el repositorio
git clone https://github.com/levihsu/OOTDiffusion.git
cd OOTDiffusion

# Crear y activar entorno conda
conda create -n ootd python==3.10 -y
conda activate ootd

# Instalar PyTorch y dependencias específicas
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Instalar requirements
pip install -r requirements.txt
```

### Paso 3: Verificar instalación inicial

```bash
# Verificar que todo está instalado correctamente
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📦 Descarga de Modelos

**Fuente de los modelos**: [levihsu/OOTDiffusion en Hugging Face](https://huggingface.co/levihsu/OOTDiffusion)

### Preparar estructura de directorios

```bash
# Instalar git-lfs
git lfs install

# Crear estructura de directorios
mkdir -p checkpoints/{ootd,humanparsing,openpose}
```

### Descargar modelos específicos

```bash
# Modelos de human parsing
wget -O checkpoints/humanparsing/parsing_atr.onnx https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx
wget -O checkpoints/humanparsing/parsing_lip.onnx https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx
wget -O checkpoints/humanparsing/exp-schp-201908261155-lip.pth https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth

# Modelos principales de OOTD (pipelines completos)
git clone https://huggingface.co/levihsu/OOTDiffusion temp_ootd
cp -r temp_ootd/checkpoints/ootd checkpoints/
cp -r temp_ootd/checkpoints/openpose checkpoints/
rm -rf temp_ootd
```

### Archivos de modelos descargados

#### Human Parsing Models:

- `parsing_atr.onnx`: Modelo ONNX para human parsing ATR
- `parsing_lip.onnx`: Modelo ONNX para human parsing LIP
- `exp-schp-201908261155-lip.pth`: Modelo PyTorch para SCHP

#### OOTDiffusion Models:

- `checkpoints/ootd/`: Pipeline completo del modelo principal
- `checkpoints/openpose/`: Modelos para detección de poses

#### CLIP Models:

- `clip-vit-large-patch14/`: Modelo CLIP para codificación de imágenes y texto

## ⚙️ Configuración del Entorno

### Solución de incompatibilidades de versiones

Si encuentras el error de importación con `huggingface_hub`, ejecuta:

```bash
# Asegúrate de estar en el entorno correcto
conda activate ootd

# Instalar versión compatible
pip install huggingface_hub==0.19.4
pip install basicsr
```

### Verificar instalaciones

```bash
# Verificar versiones instaladas
pip list | grep -E "(huggingface|diffusers|transformers)"

# Verificar que las importaciones funcionen
python -c "from huggingface_hub import hf_hub_download; print('huggingface_hub OK')"
python -c "import diffusers; print('diffusers OK')"
```

## 🎯 Uso Básico

### Preparar imágenes de prueba

```bash
# Crear estructura para imágenes de prueba
mkdir -p img_test/{clothe,person}

# Estructura esperada:
# img_test/
# ├── clothe/
# │   ├── 01260_00.jpg
# │   └── 01430_00.jpg
# └── person/
#     ├── 00891_00.jpg
#     └── 03615_00.jpg
```

### Ejecutar OOTDiffusion

```bash
# Ir al directorio de ejecución
cd OOTDiffusion/run

# Crear directorio de salida
mkdir -p images_output

# Half-body model (modelo de medio cuerpo)
python run_ootd.py \
    --model_path "../img_test/person/00891_00.jpg" \
    --cloth_path "../img_test/clothe/01260_00.jpg" \
    --model_type "hd" \
    --category "0" \
    --scale 2.0 \
    --sample 4

# Full-body model (modelo de cuerpo completo)
python run_ootd.py \
    --model_path "../img_test/person/00891_00.jpg" \
    --cloth_path "../img_test/clothe/01260_00.jpg" \
    --model_type "dc" \
    --category "1" \
    --scale 2.0 \
    --sample 4
```

### Parámetros disponibles:

- `--model_type`: `"hd"` (half-body) o `"dc"` (full-body)
- `--category`: `"0"` (upper-body), `"1"` (lower-body), `"2"` (dresses)
- `--scale`: Factor de escala (1.0-3.0)
- `--sample`: Número de imágenes a generar (1-4)

## 🛠️ Backend API Server

El proyecto incluye un servidor backend completo con detección automática de personas y métricas de rendimiento.

### Estructura del Backend

```
backend/
├── vto_server.py              # Servidor principal FastAPI
├── mobilenet_detector.py      # Detector de personas MobileNet-SSD
├── models/                    # Modelos de detección
├── temp/                      # Archivos temporales
└── clothe/                    # Catálogo de prendas
```

### Instalación del Backend

```bash
# Instalar dependencias adicionales del backend
pip install fastapi uvicorn python-multipart opencv-python

# Crear directorios necesarios
mkdir -p backend/{temp,clothe,models}

# Descargar modelo MobileNet-SSD
cd backend/models
wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt
cd ../..
```

### Configurar Catálogo de Prendas

```bash
# Agregar prendas al catálogo
cp tu_ropa_1.jpg backend/clothe/shirt_1.jpg
cp tu_ropa_2.jpg backend/clothe/dress_2.jpg
cp tu_ropa_3.jpg backend/clothe/pants_3.jpg

# La detección de categoría es automática basada en el nombre del archivo
```

### Ejecutar el Servidor Backend

```bash
# Activar entorno
conda activate ootd

# Ir al directorio backend
cd backend

# Iniciar servidor
python vto_server.py
```

El servidor estará disponible en:
- **API**: `http://localhost:8384`
- **Documentación**: `http://localhost:8384/docs`
- **Métricas**: `http://localhost:8384/metrics`

### Endpoints Disponibles

#### `POST /vto` - Virtual Try-On Principal
```bash
curl -X POST "http://localhost:8384/vto" \
     -H "Content-Type: multipart/form-data" \
     -F "person_image=@persona.jpg" \
     -F "clothe_id=1"
```

#### `GET /clothes` - Lista de Prendas Disponibles
```bash
curl "http://localhost:8384/clothes"
```

#### `GET /metrics` - Métricas de Rendimiento
```bash
curl "http://localhost:8384/metrics"
```

#### `GET /health` - Estado del Servidor
```bash
curl "http://localhost:8384/health"
```

### Respuesta de Ejemplo

```json
{
  "success": true,
  "image_url": "/result/out_hd_1234567890.png",
  "processing_time": 32.5,
  "clothe_id": 1,
  "clothe_name": "shirt_1.jpg",
  "category": "upperbody",
  "model_type": "hd",
  "metrics": {
    "average_time": 31.2,
    "category_average": 30.8,
    "recent_performance": {
      "count": 5,
      "average": 32.1,
      "min": 29.8,
      "max": 34.2
    }
  }
}
```

### Características del Backend

- **Detección automática de personas** con MobileNet-SSD
- **Clasificación automática de prendas** por nombre de archivo
- **Métricas de rendimiento** en tiempo real
- **Optimización de memoria GPU** con parámetros reducidos
- **Gestión automática de archivos temporales**
- **API RESTful completa** con documentación automática

## 🎨 Otros Modelos GAN

Este repositorio también es compatible con otros modelos de virtual try-on basados en la arquitectura de OOTDiffusion:

- **VITON-HD**: Para imágenes de alta resolución
- **HR-VITON**: Modelo de alta resolución con mejor calidad
- **CP-VTON**: Classic virtual try-on approach

### Configuración de Modelos Adicionales

```bash
# Descargar modelos adicionales
mkdir -p checkpoints/viton-hd
wget -O checkpoints/viton-hd/model.pth [URL_DEL_MODELO]
```

## 📈 Rendimiento y Optimización

### Configuración Recomendada por Hardware

#### GPU de 8GB VRAM:
```bash
python run_ootd.py --scale 1.5 --sample 2
```

#### GPU de 16GB+ VRAM:
```bash
python run_ootd.py --scale 2.0 --sample 4
```

#### CPU Only (no recomendado):
```bash
export CUDA_VISIBLE_DEVICES=""
python run_ootd.py --scale 1.0 --sample 1
```

## 📝 Créditos y Licencia

### Trabajo Original

Este proyecto está basado en **OOTDiffusion** desarrollado por:

- **Repositorio Original**: [levihsu/OOTDiffusion](https://github.com/levihsu/OOTDiffusion)
- **Paper**: "Outfit Anyone: Ultra-high quality virtual try-on for Any Clothing and Any Person"
- **Autores**: Yuhao Xu, Tao Gu, Weifeng Chen, Chengcai Chen
- **Hugging Face**: [levihsu/OOTDiffusion](https://huggingface.co/levihsu/OOTDiffusion)

### Implementación del Backend

La implementación del backend y servidor API es una extensión del trabajo original que añade:

- API RESTful para integración con aplicaciones web
- Sistema de métricas y monitoreo de rendimiento  
- Detección automática de personas con MobileNet-SSD
- Gestión automática de archivos y optimización de memoria

### Licencias

- **OOTDiffusion**: Sujeto a la licencia del repositorio original
- **Modelos Pre-entrenados**: Sujetos a sus respectivas licencias de Hugging Face
- **Extensiones del Backend**: MIT License

### Agradecimientos

Agradecemos especialmente a:

- **Equipo OOTDiffusion** por el desarrollo del modelo base y la investigación
- **Hugging Face** por el hosting de modelos y facilidades de distribución
- **Comunidad Open Source** por las contribuciones a las dependencias utilizadas

### Cita del Trabajo Original

Si utilizas este código en investigación académica, por favor cita el trabajo original:

```bibtex
@article{xu2024ootdiffusion,
  title={Outfit Anyone: Ultra-high quality virtual try-on for Any Clothing and Any Person},
  author={Xu, Yuhao and Gu, Tao and Chen, Weifeng and Chen, Chengcai},
  journal={arXiv preprint arXiv:2407.16224},
  year={2024}
}
```

---

**Implementación con fines educativos y de investigación basada en OOTDiffusion** 🎓👗
