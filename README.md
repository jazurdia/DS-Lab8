# DS-Lab8
## Informe de Reflexión - Puesta en Producción de un Modelo de ML

Realizado por:
- Javier Alejandro Azurdia Arrecis 21242
- Diego Alejandro Morales Escobar 21146
- Angel Sebastian Castellanos Pineda 21700

### Desafíos durante la selección del modelo y puesta en producción  
Uno de los mayores desafíos fue identificar el modelo más adecuado para los datos proporcionados. Implementamos tres algoritmos: Regresión Lineal, Random Forest y Gradient Boosting. Aunque los modelos de ensamble suelen ser más robustos para relaciones complejas, descubrimos que la Regresión Lineal superaba a los otros en este caso, debido a que los datos presentaban relaciones mayormente lineales. Además, la limpieza y preprocesamiento del dataset representaron un reto importante, especialmente al manejar valores atípicos y transformar variables categóricas mediante *one-hot encoding* para asegurar compatibilidad con los modelos.

El despliegue del modelo también presentó complicaciones. La integración con Streamlit para crear una interfaz interactiva exigió ajustes cuidadosos para alinear los datos ingresados por el usuario con las columnas esperadas por el modelo. Asegurar que la aplicación cargara correctamente el modelo entrenado y manejar errores de entrada fueron aspectos críticos.

### Aprendizajes más significativos  
Durante este laboratorio, reforzamos la importancia del análisis exploratorio de datos para identificar patrones y posibles problemas que puedan afectar el rendimiento del modelo. Además, la experiencia con Streamlit nos enseñó cómo simplificar la interacción del usuario con modelos complejos, lo cual es crucial para implementar soluciones de *Machine Learning* en escenarios reales.

### Evaluación crítica del enfoque utilizado  
El enfoque de usar varios modelos permitió identificar rápidamente la opción más eficiente para los datos, destacando la simplicidad y precisión de la Regresión Lineal en este caso. Sin embargo, una limitación fue la falta de exploración más profunda de técnicas de hiperparametrización en los modelos de ensamble, lo que podría haber mejorado su rendimiento. También notamos que el dataset tenía ciertas limitaciones en cuanto a variedad, lo que puede afectar la generalización del modelo.

### Sugerencias para mejorar futuros procesos  
Para optimizar el proceso de desarrollo y despliegue, recomendamos:
1. **Automatizar el preprocesamiento**: Utilizar pipelines de datos para evitar errores manuales y asegurar consistencia en cada ejecución.
2. **Explorar más algoritmos**: Implementar otros modelos como *XGBoost* o *LightGBM* para tener más opciones robustas.
3. **Implementar validación continua**: Integrar técnicas de monitoreo del modelo en producción para detectar cambios en el comportamiento de los datos.
4. **Hiperparametrización más exhaustiva**: Usar *GridSearchCV* o *RandomSearchCV* para afinar los modelos de manera más efectiva.
