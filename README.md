# Segmentación de clientes con K-Means

**Autor:** Juan Felipe Mora Revelo
**Asignatura:** Inteligencia de Negocios y Minería de Datos
**Institución:** Universidad Cooperativa de Colombia

Segmentación de los 200 clientes del dataset *Mall Customers* (Kaggle) con K-Means, evaluada con el método del codo y el coeficiente de silueta, e interpretada desde la perspectiva de marketing. Incluye una aplicación interactiva en Streamlit.

## Contenido del proyecto

| Archivo | Qué es |
|---|---|
| `Taller_KMeans_Segmentacion.ipynb` | Cuaderno de Jupyter con las Partes 1 a 6: código, gráficas y respuestas a cada pregunta |
| `app.py` | Aplicación de Streamlit (Parte 7) |
| `Mall_Customers.csv` | Dataset original |
| `clientes_segmentados.csv` | Dataset con el cluster y perfil de cada cliente (lo genera el cuaderno) |
| `modelo_kmeans.pkl` | Modelo y escalador entrenados (los genera el cuaderno) |
| `requirements.txt` | Librerías necesarias |
| `.streamlit/config.toml` | Colores de la app en modo claro y oscuro |

## Cómo ejecutarlo

1. Instalar las librerías (una sola vez), desde una terminal abierta en esta carpeta:
   ```bash
   pip install -r requirements.txt
   ```
2. Cuaderno: abrir `Taller_KMeans_Segmentacion.ipynb` en Jupyter o VS Code y ejecutar todas las celdas (*Run All*).
3. Aplicación:
   ```bash
   streamlit run app.py
   ```
   Se abre sola en el navegador en `http://localhost:8501`.

## Resultados principales

- Variables de segmentación: **ingreso anual** y **puntaje de gasto**, escaladas con `StandardScaler`.
- **K óptimo = 5**: el método del codo y el coeficiente de silueta (**0.555**) coinciden.
- Perfiles: clientes de alto valor, con potencial, promedio, impulsivos y de bajo valor.
- Mayor oportunidad de crecimiento: **clientes con potencial** (ingreso alto, gasto bajo).

## Funcionalidades de la app

- Exploración de datos con gráficas interactivas.
- Método del codo y coeficiente de silueta (con diagrama de silueta por cliente).
- Clusters en 2D y 3D, centroides, perfiles y estrategias de marketing.
- Formulario para ingresar un cliente nuevo y clasificarlo automáticamente, con historial descargable en CSV.
- Modo claro y oscuro, y diseño adaptable a celular.
- Selector de K en la barra lateral para comparar otros valores.
