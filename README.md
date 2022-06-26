# Detector de rostros 'nico-not_nico' en tiempo real

Detectector de rostros en tiempo real usando Python, TensoFlow/Keras y OpenCV. 

Este programa difenrenica una persona del resto meidante la cámara web.

## Requisitos

Estos códigos necesitan las librerías *tensorflow* y *opencv*. La forma más fácil de conseguir esto es ejecutar el comando `pip install -r requirements.txt`.

## Estructura del proyecto

La estructura es la siguiente

```
nico_not_nico_detector
├───datasets
│   ├───face_dataset_test_images
│   │   ├───nico      
│   │   └───not_nico  
│   ├───face_dataset_train_aug_images
│   │   ├───nico 
│   │   └───not_nico  
│   └───face_dataset_train_images
│       ├───nico      
│       └───not_nico  class
├───models
│   .gitignore
│   data_augmentation.py
│   nico_not_nico_classifier.py
│   nico_not_nico_classifier_model_comparison.ipynb
│   nico_not_nico_detector.py
│   README.md
└── requirements.txt
```

Los directorios del repositorio son los siguientes:
- El directorio `models` contiene los modelos entrenados previamente
- El directorio `datasets` contiene a su vez tres directorios, para el entrenamiento, para el entrenamiento con cantidad de datos aumentados y para validación. En cad uno de estos directorios hay a suz vez dos directorios que se corresponden a las clases que queremos identificar, *nico* y *not_nico*.

En cuanto a los archivos de código:
- el archivo `data_augmentation.py` aumenta de forma artificial el conjunto de datos original.
- el cuaderno `nico_not_nico_classifier_model_comparison.ipynb` contiene el cogio necesario para entrenar y evaluar cinco modelos diferentes.
- el archivo `nico_not_nico_classifier.py` construye un modelo específico.
- el archivo `nico_not_nico_detector.py` usa OpenCV para converitr el modelo en un clasificador de rostros en tiempo real.

# Referencias
Este código se basa en el proyecto *me_not_me_detector* del autor Dmytro Nikolaiev, que puede consultarse en el siguiente enlace: https://gitlab.com/Winston-90/me_not_me_detector/