import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
#from skimpy import skim
#import missingno as msno
from scipy.stats import shapiro, gaussian_kde
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


#charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("E:\Apprentissage\PROJET\Projet1\creditcard.csv\creditcard.csv")
    return df

#Description du projet
def introduction():
    st.title("Analyse Exploratoire des Données pour la détection des fraudes bancaires")
    st.subheader("""
                 Auteur: [Ernest Dave](https://www.linkedin.com/in/ernest-dave-aounang-mi-njampou-a318a31a6) & [Mamadou Oury Balde](https://www.linkedin.com/in/mamadou-oury-balde-4270301ab/)
                 """)
    st.markdown("""
    Ce tableau de bord présente les résultats de l'analyse exploratoire des données
    sur un ensemble de transactions bancaires pour identifier des schémas de fraude.
    Les données contiennent des informations sur les transactions par carte de crédit,
    avec un faible taux de fraude (0.172% des transactions).
    """)

#Profilage des données
def show_data_profile(df):
    st.subheader("Profilage des Données")
    st.write("Nombre total de transactions: ", df.shape[0])
    st.write("Nombre total de variables: ", df.shape[1])

    #Afficher les 5 premières lignes
    st.write("Echantillon des données")
    st.write(df.head())

    #Afficher les informations sur les variables
    st.write("Informations sur les variables")
    st.write(df.info())
    st.write("""Toutes les variables caractéristiques (features) sont de type float64, ce qui indique qu'elles représentent des données quantitatives continues. La variable cible, Class, est de type int64, une donnée qualitative nominale.

    - ** Time ** : Nombre de secondes écoulées depuis la première transaction enregistrée jusqu'à la transaction en cours.
    - ** V1 à V28 ** : Variables dérivées d'une réduction de dimensionnalité par l'algorithme PCA (Principal Component Analysis). Ces variables ont été anonymisées pour préserver la confidentialité des clients.
    - ** Amount ** : Montant de la transaction.
    - ** Class ** : Variable cible indiquant deux classes : 0 pour une transaction normale et 1 pour une transaction frauduleuse.
    """)

#Analyse univariée des vraiables
def show_univariate_analysis(df):
    st.subheader("Analyse Univariée")

    # Sélection des colonnes et paramètres
    columns = df.select_dtypes(exclude=['int64']).columns.tolist()
    selected_column = st.selectbox("Sélectionner une variable", columns)
    graph_type = st.selectbox('Choisir un type de graphique', ['Histogramme et Boxplot', 'KDE et Boxplot'])

    n_bins = st.number_input('Choisir un nombre de classes (bins)', min_value=10, max_value=100, value=50)
    graph_title = f"Analyse de la variable {selected_column}"

    
    if graph_type == 'Histogramme et Boxplot':
        # Créer une figure avec deux graphiques côte à côte
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogramme", "Boxplot"))

        # Ajouter l'histogramme
        fig.add_trace(
            go.Histogram(
                x=df[selected_column],
                nbinsx=n_bins,
                marker=dict(color='mediumslateblue', line=dict(color='black', width=1)),
                opacity=0.75,
                name=selected_column
            ),
            row=1, col=1
        )

        # Ajouter le boxplot
        fig.add_trace(
            go.Box(
                y=df[selected_column],
                name=selected_column,
                marker_color='mediumslateblue'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=graph_title,
            xaxis_title=selected_column,
            template='plotly_white'
        )
        st.plotly_chart(fig)

    elif graph_type == 'KDE et Boxplot':
        # Créer une figure avec deux graphiques côte à côte
        fig = make_subplots(rows=1, cols=2, subplot_titles=("KDE", "Boxplot"))

        # Calcul des densités pour KDE
        x_values = np.linspace(df[selected_column].min(), df[selected_column].max(), 100)
        kde = gaussian_kde(df[selected_column])
        kde_values = kde(x_values)

        # Ajouter la courbe KDE
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=kde_values,
                mode='lines',
                fill='tozeroy',
                line=dict(color='mediumslateblue'),
                name='KDE'
            ),
            row=1, col=1
        )

        # Ajouter le boxplot
        fig.add_trace(
            go.Box(
                y=df[selected_column],
                name='Boxplot',
                marker_color='mediumslateblue'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=graph_title,
            xaxis_title=selected_column,
            template='plotly_white'
        )
        st.plotly_chart(fig)


    # Calcul des statistiques
    skewness = df[selected_column].skew()
    kurtosis = df[selected_column].kurtosis()
    stat, p_value = shapiro(df[selected_column])
    st.write(f"Skewness: {skewness:.2f}")
    st.write(f"Kurtosis: {kurtosis:.2f}")
    st.write(f"Test de normalité (Shapiro-Wilk): Statistique={stat:.2f}, p-value={p_value:.2f}")


#Analyse univariée des vraiables après transformation
def show_univariate_analysis_transform(df):
    df_clean = df.copy()
    df_clean.drop('Time_categ', axis=1, inplace=True)
    show_univariate_analysis(df_clean)


#Elimination des valeurs aberrantes
@st.cache_data
def transformation(df):
    df_clean = df.copy()
    columns = df_clean.select_dtypes(exclude=['int64']).columns.tolist()

    #discrétisation de la variable Time
    bins = [0, 6, 9, 12, 18, 21, 24]
    labels = ['Nuit', 'Tôt le matin', 'Matinée', 'Après-midi', 'Soirée', 'Nuit tardive']
    df_clean['Time_categ'] = pd.cut((df_clean['Time']/3600) % 24, bins=bins, labels=labels, right=False)
    df_clean.drop('Time', axis=1, inplace=True)

    #transformation logarithmique des autres variables
    columns.remove('Time')
    for i in columns:
        df_clean[i] = df[i].apply(lambda x: np.sign(x) * np.log(np.abs(x) + 1))
    
    return df_clean

#Matrice de corrélation
def show_correlation_matrix(df):
    # Sélection des colonnes sauf 'Time_categ'
    col = df.columns.tolist()
    col.remove('Time_categ')
    
    # Calcul de la matrice de corrélation
    st.subheader("Matrice de Corrélation")
    corr = df[col].corr(method='spearman')
    corr = corr.round(2)
    text_values = corr.applymap(lambda x: f"{x:.2f}")

    # Création de la heatmap avec Plotly
    # fig = go.Figure(data=go.Heatmap(
    #     z=corr.values,  
    #     x=corr.columns,  
    #     y=corr.columns, 
    #     colorscale='Blues',
    #     zmin=-1, zmax=1,  
    #     colorbar=dict(title="Corrélation"),  
    #     # text=text_values.values,  
    #     hovertemplate="Corrélation: %{text} entre %{x} et %{y}<extra></extra>", 
    #     text_auto=True,
    # ))

    fig = px.imshow(corr.values, x=corr.columns, y=corr.columns, color_continuous_scale='Blues', aspect="auto")
    fig.update_traces(text=text_values, texttemplate="%{text}")
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title="Matrice de Corrélation",
        title_x=0.5,  
        height=800,   
        width=1000,
        template='plotly_white',
        # xaxis_title="Variables",
        # yaxis_title="Variables",
    )
    st.plotly_chart(fig)

#Visualisation des relations
def show_relationships(df):
    st.subheader("Visualisation des Relations")
    columns = df.select_dtypes(exclude=['int64']).columns.tolist()
    columns.remove('Time_categ')
    df['class_str']=df['Class'].astype(str)
    x = st.selectbox("Sélectionner une variable pour l'axe x", columns)
    y = st.selectbox("Sélectionner une autre variable pour l'axe y", columns)

    fig2 = px.scatter(
        data_frame=df,
        x=x,
        y=y,
        title=f'{x} VS {y}',
        color='class_str',  
        labels={x: x, y: y}, 
        symbol='class_str',  
    )

    st.plotly_chart(fig2)

#Analyse des fraudes par tranches horaires
def analyse(df):
    st.subheader("Analyse des Fraudes Par Tranches Horaires")
    
    # Labels pour les tranches horaires
    lab = ['Nuit - 0-6h', 'Tôt le matin - 6-9h', 'Matinée - 9-12h', 'Après-midi - 12-18h', 'Soirée - 18-21h', 'Nuit tardive - 21-24h']

    # Spécification des types de subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=(
            "Transactions normales par tranche horaire",
            "Transactions frauduleuses par tranche horaire",
            "Transactions normales par tranche horaire",
            "Transactions frauduleuses par tranche horaire"
        ),
        horizontal_spacing=0.25,  
        vertical_spacing=0.2     
    )

    # Diagramme en barres pour les transactions normales
    fig.add_trace(
        go.Bar(
            x=lab,
            y=df[df['Class'] == 0].groupby('Time_categ').size().values,
            name='Normal', marker_color='mediumslateblue',
            legendgroup='bar_group'
        ),
        row=1, col=1
    )

    # Diagramme en barres pour les transactions frauduleuses
    fig.add_trace(
        go.Bar(
            x=lab,
            y=df[df['Class'] == 1].groupby('Time_categ').size().values,
            name='Fraud', marker_color='red',
            legendgroup='bar_group'
        ),
        row=1, col=2
    )
    
    # Diagramme en camembert pour les transactions normales
    fig.add_trace(
        go.Pie(
            labels=lab,
            values=df[df['Class'] == 0].groupby('Time_categ').size().values,
            name='Normal', hole=0.3,
            legendgroup='pie_group'
        ),
        row=2, col=1
    )

    # Diagramme en camembert pour les transactions frauduleuses
    fig.add_trace(
        go.Pie(
            labels=lab,
            values=df[df['Class'] == 1].groupby('Time_categ').size().values,
            name='Fraud', hole=0.3,
            legendgroup='pie_group'
        ),
        row=2, col=2
    )

    # Mise à jour du layout
    fig.update_layout(
        template='plotly_white',
        title="Analyse des Fraudes Par Tranches Horaires",
        title_x=0.5,  
        height=800,   
        width=1000,
        showlegend=True,
        # legend=dict(
        #     x=1.05,  
        #     y=0.5,   
        #     traceorder='normal',
        #     orientation='v',
        #     font=dict(size=12),
        #     bgcolor='rgba(255, 255, 255, 0)', 
        # ),
       
    )
    
    st.plotly_chart(fig)

#Test de significativité
def test_significativite(df):
    results = {
        "Test": [],
        "Statistique de test": [],
        "p-value": [],
        "Interprétation": []
    }

    # Test de Mann-Whitney U
    columns = df.select_dtypes(exclude=['int64']).columns.tolist()
    columns.remove('Time_categ')

    st.write("# Test de Significativité")
    st.subheader("Association entre les variables quantitatives et la variable cible")
    st.subheader("Test de Mann-Whitney U")
    selected_column = st.selectbox("Sélectionner une variable", columns)
    stat, p = stats.mannwhitneyu(df[df['Class'] == 0][selected_column], df[df['Class'] == 1][selected_column])
    
    # Ajout des résultats du test de Mann-Whitney U
    results["Test"].append("Mann-Whitney U")
    results["Statistique de test"].append(f"{stat:.2f}")
    results["p-value"].append(f"{p:.2f}")
    if p < 0.05:
        results["Interprétation"].append("Il y a une différence significative entre les deux groupes")
    else:
        results["Interprétation"].append("Il n'y a pas de différence significative entre les deux groupes")

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Test de Chi2
    st.subheader("Association entre la variable Time_Categ et la variable cible")
    st.subheader("Test du Chi2 & Coefficient de Tschuprow")
    tab = pd.crosstab(df['Time_categ'], df['Class'])
    stat, p, dof, expected = stats.chi2_contingency(tab)
    
    # Ajout des résultats du test de Chi2
    results["Test"].append("Chi2")
    results["Statistique de test"].append(f"{stat:.2f}")
    results["p-value"].append(f"{p:.2f}")
    if p < 0.05:
        results["Interprétation"].append("Il y a une différence significative entre les deux groupes")
    else:
        results["Interprétation"].append("Il n'y a pas de différence significative entre les deux groupes")

    # Coefficient de Tschuprow
    n = tab.sum().sum()
    r, c = tab.shape
    coef = np.sqrt(stat/(n*min(r-1, c-1)))
    
    # Ajout des résultats du coefficient de Tschuprow
    results["Test"].append("Coefficient de Tschuprow")
    results["Statistique de test"].append(f"{coef:.2f}")
    results["p-value"].append("")
    if coef > 0.5:
        results["Interprétation"].append("Il y a une association forte entre les deux variables")
    else:
        results["Interprétation"].append("Il n'y a pas d'association forte entre les deux variables")
    
    results_df = pd.DataFrame(results)
    results_df.drop(index=0, inplace=True)
    st.dataframe(results_df)


#Mise en page du tableau de bord
def main():
    st.sidebar.title('Menu')
    options = ['Introduction', 'Profilage des Données', 'Analyse Univariée', 'Analyse Univariée avec Log-Modulus', 'Matrice de Corrélation', 
               'Visualisation des Relations', 'Analyse des Fraudes', 'Test de Significativité']
    choice = st.sidebar.selectbox("Choisir une section", options)
    
    df = load_data()
    df_clean = transformation(df)

    if choice == 'Introduction':
        introduction()
    elif choice == 'Profilage des Données':
        show_data_profile(df)
    elif choice == 'Analyse Univariée':
        show_univariate_analysis(df)
    elif choice == 'Analyse Univariée avec Log-Modulus':
        show_univariate_analysis_transform(df_clean)
    elif choice == 'Matrice de Corrélation':
        show_correlation_matrix(df_clean)
    elif choice == 'Visualisation des Relations':
        show_relationships(df_clean)
    elif choice == 'Analyse des Fraudes':
        analyse(df_clean)
    elif choice == 'Test de Significativité':
        test_significativite(df_clean)  

# Exécution du tableau de bord
if __name__ == '__main__':
    main()


