# Automated Biological Data (EDA) Exploratory Data Analysis Streamlit App
🧬 Automated Biological Data EDA (Exploratory Data Analysis) (Streamlit App). Explore the generated plots and insights interactively.

This project provides an interactive **Streamlit app** for automated exploratory data analysis (EDA) of biological datasets.  
It allows users to upload CSV files (or use synthetic sample data) and generates **interactive visualizations** to quickly understand patterns, distributions, and relationships.


## ✨ Features
- Upload your own CSV or use built-in synthetic gene expression data.
- Data preview with shape, columns, and missing values.
- **Univariate analysis:**
  - Histograms with box overlays
  - Bar charts for categorical variables
- **Bivariate analysis:**
  - Scatter plots (with optional grouping)
  - Box plots grouped by categories
- **Additional plots:**
  - Pair plots (Seaborn)
  - Violin plots
  - Kernel Density Estimation (KDE) plots
- **Correlation heatmap** with Pearson, Spearman, or Kendall methods.
- **Missingness overview** across all columns.


**Create a virtual environment (recommended)**
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\\Scripts\\activate    # On Windows


**Install dependencies**
pip install -r requirements.txt
streamlit
pandas
numpy
plotly
seaborn
matplotlib


**📊 Usage**

Upload your biological dataset as a CSV (e.g., gene expression, proteomics, metabolomics).

Use the sidebar to configure:

Delimiter and encoding

Sampling size

Log-transform (for expression counts)

Explore the generated plots and insights interactively.


<img width="1673" height="485" alt="image" src="https://github.com/user-attachments/assets/234ca59b-5e22-4b90-8006-1077165030be" />


<img width="1200" height="550" alt="image" src="https://github.com/user-attachments/assets/c94d4dfb-32dd-49b6-8bd0-01fd071704ab" />




https://github.com/user-attachments/assets/2bcdc2ba-5fc9-4491-95b6-93ad1add54f6

