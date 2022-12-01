# Detector de rostros 'Nico-Not_Nico' en tiempo real

## Introducción
Detector de rostros en tiempo real entrenado para identificar una persona en específico, que no es otra que el autor de este proyecto, empleando para ello Python, Tensorflow/Keras y OpenCV.

## Estructura del proyecto

La estructura es la siguiente

```
nico_not_nico_detector
├───datasets
│   ├───dataset_train
│   │   ├───nico 
│   │   └───not_nico  
│   └───dataset_validation
│       ├───nico      
│       └───not_nico 
├───logs
│   └───fit
├───models
│   .gitignore
│   nico_not_nico_classifier.py
│   nico_not_nico_classifier_fine_tune.py
│   nico_not_nico_detector.py
│   README.md
└── requirements.txt
```

Los directorios del repositorio son los siguientes:
- El directorio `models` contiene los modelos entrenados previamente
- El directorio `datasets` contiene a su vez dos directorios, para el entrenamiento y para validación. En cad uno de estos directorios hay a suz vez dos directorios que se corresponden a las clases que queremos identificar, *nico* y *not_nico*.
- El directorio `logs/fit` contiene la información registrada durante el entrenamiento para poder ser visualizada en `tensorboard`.

## Requisitos
Los requereimientos para este proyecto se encuentran listados en el archivo ```requirements.txt```, para instalarlos puede ejecutar la siguiente instrucción:

```sh
pip install -r requirements.txt
```

## Entrenar modelo
El entrenamiento del modelo emplea una red pre-entranada para el reconocimiento facial, para cambiar entre diferentes redes, deben editarse en el archivo ```nico_not_nico_classifier.py``` los siguientes campos:
- Modelo base
- Preprocesado de reescalado
- Nombre del modelo (opcional)

Pueden encontrarse en forma de comentario las opciones para las redes MobileNet (por defecto), ResNet50, ResNet152, Xception y VGG16.

El entrenamiento puede ejecutarse con:
```sh
python3 nico_not_nico_classifier.py
```

El archivo ```nico_not_nico_classifier_fine_tune.py``` es similar, pero ejecuta dos etapas de entrenamiento, una primera normal y una segunda en la que se descongelan las capas superiores del modelo base.

```sh
python3 nico_not_nico_classifier_fine_tune.py
```

## Ejecutar detector
El detector emplea OpenCV para detectar rostros y realizar las inferencias sobre éstos empleando el modelo correspondiente. Se ejecuta con:
```sh
python3 nico_not_nico_detector.py
```  

## Tensorboard
Para examinar los detalles de la creación del modelo puede emplearse la herramienta ``Tensorboard`` con la siguiente instrucción:
```sh
tensorboard --logdir logs/fit
```

## Referencias
- [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [Common architectures in convolutional neural networks.](https://www.jeremyjordan.me/convnet-architectures/)
- [Image Augmentation for Deep Learning](https://towardsdatascience.com/image-augmentation-for-deep-learning-histogram-equalization-a71387f609b2)
- [How to Create a Real-Time Face Detector](https://towardsdatascience.com/how-to-create-real-time-face-detector-ff0e1f81925f)
- [How to Perform Face Recognition With VGGFace2 in Keras](https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/)