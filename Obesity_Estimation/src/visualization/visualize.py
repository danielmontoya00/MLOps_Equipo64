import matplotlib.pyplot as plt
import seaborn as sns
import os

FIG_PATH = "../reports/figures"
os.makedirs(FIG_PATH, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(f"{FIG_PATH}/{filename}", bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_numeric_distributions(df, numeric_cols):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribución de {col}')
        save_plot(fig, f"{col}_hist.png")

def plot_categorical_distributions(df, categorical_cols):
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette="Set2", ax=ax)
        ax.set_title(f'Distribución de {col}')
        plt.xticks(rotation=45)
        save_plot(fig, f"{col}_count.png")

def plot_correlation_matrix(df, numeric_cols):
    corr = df[numeric_cols].corr(method='pearson')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Matriz de Correlación (Pearson)')
    save_plot(fig, "corr_matrix.png")

def plot_bmi_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='IMC_category',
                  order=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad III'],
                  palette="Set2", ax=ax)
    ax.set_title("Distribución por categoría de IMC (OMS)")
    plt.xticks(rotation=45)
    save_plot(fig, "imc_distribution.png")