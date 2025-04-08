#!/bin/bash

# evaluate_model.sh - Script para evaluar modelos entrenados de detección de mentiras

# Configuración básica
DEVICE="cuda:0"                  # Dispositivo para evaluación (cuda:0, cpu)
MODEL_TYPE="audio"              # Tipo de modelo (audio, vision, fusion)
MODEL_PATH="best_model_audio.pth"  # Ruta al modelo guardado
TEST_FILE="test_fold3.csv"       # Archivo CSV de prueba
BATCH_SIZE=16                    # Tamaño del lote
DATA_ROOT="C:/Users/victo/deceptionDetector/dataset/DOLOS/"  # Directorio raíz de datos

# Rutas específicas (ajustar según necesidad)
AUDIO_PATH="C:/Users/victo/deceptionDetector/dataset/audio_files/"
VISUAL_PATH="C:/Users/victo/deceptionDetector/dataset/face_frames/"

# Parámetros del modelo (deben coincidir con los de entrenamiento)
NUM_ENCODERS=4                   # Número de codificadores transformer
ADAPTER=true                     # Usar adaptadores (true/false)
ADAPTER_TYPE="efficient_conv"    # Tipo de adaptador
FUSION_TYPE="cross2"             # Tipo de fusión (solo para modelo fusion)

# Crear directorio para resultados si no existe
mkdir -p results

# Nombre del archivo de resultados (basado en el archivo de prueba)
RESULT_FILE="results/eval_results__${MODEL_TYPE}.txt"

# Comando de evaluación
python test_model.py \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --data_root $DATA_ROOT \
  --audio_path $AUDIO_PATH \
  --visual_path $VISUAL_PATH \
  --test_file $TEST_FILE \
  --model_to_train $MODEL_TYPE \
  --model_path $MODEL_PATH \
  --num_encoders $NUM_ENCODERS \
  ${ADAPTER:+--adapter} \
  --adapter_type $ADAPTER_TYPE \
  --fusion_type $FUSION_TYPE \
  ${MULTI:+--multi} \
  | tee $RESULT_FILE

echo "Evaluación completada. Resultados guardados en $RESULT_FILE"