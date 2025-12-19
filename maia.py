import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ----- Configurare pagina -----

st.set_page_config(
    page_title="Dashboard Analiză Date",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- CSS -----

st.markdown(
"""
<style>
/* Background principal */
.stApp {
background: linear-gradient(135deg, #1b102a, #2b1745);
color: #EDE7F6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
background-color: #1a0f2e;
border-right: 1px solid #3d2a5d;
}

/* Titluri */
h1, h2, h3 {
color: #E1BEE7;
font-weight: 700;
}

/* Text normal */
p, label, div {
color: #EDE7F6;
}

/* Butoane */
button {
background: linear-gradient(90deg, #7B1FA2, #9C27B0);
color: white;
border-radius: 12px;
padding: 0.5rem 1rem;
border: none;
font-weight: 600;
}

button:hover {
background: linear-gradient(90deg, #9C27B0, #BA68C8);
}

/* Metric */
div[data-testid="stMetric"] {
background-color: #24123a;
padding: 1rem;
border-radius: 16px;
border: 1px solid #3d2a5d;
}

/* Input fields */
input, textarea {
background-color: #2b1745 !important;
color: #EDE7F6 !important;
border-radius: 10px !important;
}

/* Ascunde footer Streamlit */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""",

unsafe_allow_html=True
)

# ----- Navigare -----
def load_data_from_csv(uploaded_file):
    """Încarcă datele dintr-un fișier CSV încărcat în Streamlit."""
    df = pd.read_csv(uploaded_file,on_bad_lines='skip')
    return df

def show_data_reading():
    st.title("Aplicatie de Analiza a Datelor")
    st.caption("Template Streamlit • Teodorescu Ingrid • Python")

    st.markdown("#### Aceasta aplicatie este o platforma interactiva de analiza si prelucrare a volumelor mari de date")

    st.markdown("---")

    st.markdown("## Încărcare fișier CSV/Excel")

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_file = st.file_uploader(
            "**Încarcă fișier CSV/Excel**",
            type=["csv", "xlsx"]
        )

    with col2:
        st.markdown("**Nu ai date?**")
        sample_df = pd.read_csv("petfinder-mini.csv")  # fișier local
        st.download_button(
            label="⬇️ Descarcă set exemplu",
            data=sample_df.to_csv(index=False),
            file_name="petfinder-mini.csv",
            mime="csv"
        )
        st.markdown("Descarcă set exemplu si incarca acest fisier csv")

    tab_info, tab_filter = st.tabs(["Sample tabel", "Filtre"])

    with tab_info:
        if uploaded_file is not None:
            try:
                df_csv = load_data_from_csv(uploaded_file)
                st.session_state['df'] = df_csv
                st.session_state['collection_name'] = "csv_upload"

                st.success(f"Date încărcate cu succes din CSV! ({len(df_csv):,} rânduri, {len(df_csv.columns)} coloane)")

                st.dataframe(df_csv.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Eroare la citirea fisierului: {e}")
            except UnboundLocalError:
                st.warning("Vă rugăm să incarcati fișier Excel/CSV pentru a vizualiza.")
                return

            if 'df' in st.session_state:
                df = st.session_state['df']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Rânduri", f"{len(df):,}")

            with col2:
                st.metric("Total Coloane", len(df.columns))

            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric(" Memorie", f"{memory_mb:.2f} MB")

            with col4:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric(" Valori Lipsă", f"{missing_pct:.1f}%")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    with tab_filter:
        if 'df' not in st.session_state:
            st.warning("Vă rugăm să consultați secțiunea 'Încărcare, validare fișier Excel/CSV'")
            return

        n_rows = st.slider("Număr rânduri de afișat:", 5, 50, 10, key="preview_rows")
        with st.expander(f"Primele ({n_rows}) rânduri", expanded=True):
            st.dataframe(df.head(n_rows), use_container_width=True)

        col_num, col_cat = st.columns(2)

        with col_num:

            numeric_filters = {}

            for col in numeric_cols:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                numeric_filters[col] = st.slider(
                    col, min_val, max_val, (min_val, max_val)
                )

        with col_cat:

            categorical_filters = {}

            for col in categorical_cols:
                with st.expander(f"{col} ({len(df[col].unique())} opțiuni)", expanded=False):
                    options = df[col].dropna().unique().tolist()
                    categorical_filters[col] = st.multiselect(
                        label=f"Alege valori pentru {col}",
                        options=options,
                        default=options
                    )

        st.write("---")
        apply_filters = st.button("Aplică filtre")

        if apply_filters:
            df_filtered = df.copy()

            rows_before = len(df_filtered)

            for col, (min_val, max_val) in numeric_filters.items():
                df_filtered = df_filtered[
                    (df_filtered[col] >= min_val) &
                    (df_filtered[col] <= max_val)
                    ]

            for col, selected in categorical_filters.items():
                if selected:
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]

            rows_after = len(df_filtered)

            st.success("Filtrare aplicată cu succes")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rânduri inițiale", rows_before)
            with col2:
                st.metric("Rânduri după filtrare", rows_after)

            st.subheader("📄 DataFrame filtrat")
            st.dataframe(df_filtered)

def show_data_info():
    if 'df' not in st.session_state:
        st.warning("Vă rugăm să consultați secțiunea 'Încărcare, validare fișier Excel/CSV'")
        return

    df = st.session_state['df'].copy()

    st.markdown("## Informații Dataset")

    tab_info, tab_stats, tab_missing = st.tabs(["Informatii Generale", "Statistici", "Valori Lipsa"])

    with tab_info:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Distribuția Tipurilor**")
            type_counts = df.dtypes.astype(str).value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Tipuri de Date"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Tipuri de Date & Non-Null**")
            dtype_df = pd.DataFrame({
                'Coloană': df.columns,
                'Tip': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                '%Null': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(dtype_df, use_container_width=True)

    with tab_stats:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        col1, col2 = st.columns([1, 1])

        with col1:
            if numeric_cols:
                with st.expander("Coloane Numerice", expanded=True):
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        with col2:
            if categorical_cols:
                with st.expander("Coloane Categorice", expanded=True):
                    cat_summary = pd.DataFrame({
                        col: [
                            df[col].nunique(),
                            df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                            df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                            f"{(df[col].value_counts().iloc[0] / len(df) * 100):.1f}%"
                        ] for col in categorical_cols
                    }, index=['Valori Unice', 'Cel Mai Comun', 'Frecvență', 'Procent']).T
                    st.dataframe(cat_summary, use_container_width=True)

    with tab_missing:
        col1, col2 = st.columns([1, 2])
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Coloană': missing.index,
            'Număr Lipsă': missing.values,
            'Procent': missing_pct.values
        }).sort_values('Număr Lipsă', ascending=False)

        cols_with_missing = missing_df[missing_df['Număr Lipsă'] > 0]
        if len(cols_with_missing) > 0:
            with col1:

                st.markdown("**Tabel Valori Lipsă**")
                st.dataframe(cols_with_missing, use_container_width=True)

                st.markdown("### Heatmap Valori Lipsă (primele 50 rânduri)")
                colours = ['#ffff00', '#000099']  # yellow = missing, blue = present
                fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
                sns.heatmap(df.head(50).isnull(), cmap=sns.color_palette(colours),
                            cbar=False, yticklabels=False, ax=ax)
                ax.set_title("Galben = Lipsă, Albastru = Prezent")
                st.pyplot(fig, use_container_width=False)

            with col2:

                    st.markdown("**Grafic Valori Lipsă**")
                    fig = px.bar(
                        cols_with_missing,
                        x='Coloană',
                        y='Procent',
                        title='Procentul Valorilor Lipsă pe Coloană',
                        text='Număr Lipsă'
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Selectează Metoda de Tratare")

                    col_to_treat = st.selectbox(
                        "Selectează coloana de tratat:",
                        cols_with_missing['Coloană'].tolist()
                    )

                    col_type = df[col_to_treat].dtype
                    is_numeric = np.issubdtype(col_type, np.number)

                    if is_numeric:
                        strategy = st.radio(
                            "Alege strategia:",
                            ['mean', 'median', 'constant', 'interpolate', 'drop'],
                            format_func=lambda x: {
                                'mean': ' Medie',
                                'median': ' Mediană',
                                'constant': ' Valoare Constantă',
                                'interpolate': ' Interpolare',
                                'drop': ' Elimină Rânduri'
                            }[x],
                            horizontal=True
                        )

                        if strategy == 'constant':
                            fill_value = st.number_input("Valoarea de înlocuire:", value=0.0)
                        elif strategy == 'interpolate':
                            interp_method = st.selectbox(
                                "Metodă interpolare:",
                                ['linear', 'polynomial', 'spline'],
                                help="Linear = cel mai comun"
                            )
                    else:
                        strategy = st.radio(
                            "Alege strategia:",
                            ['mode', 'constant', 'drop'],
                            format_func=lambda x: {
                                'mode': 'Mod (Cel mai frecvent)',
                                'constant': ' Valoare Constantă',
                                'drop': ' Elimină Rânduri'
                            }[x],
                            horizontal=True
                        )

                        if strategy == 'constant':
                            fill_value = st.text_input("Valoarea de înlocuire:", value="_MISSING_")

                    if st.button(" Aplică ", type="primary"):
                        df_treated = df.copy()

                        try:
                            if strategy == 'mean':
                                fill_val = df_treated[col_to_treat].mean()
                                df_treated[col_to_treat].fillna(fill_val, inplace=True)
                                st.success(f" Înlocuite cu media: {fill_val:.2f}")

                            elif strategy == 'median':
                                fill_val = df_treated[col_to_treat].median()
                                df_treated[col_to_treat].fillna(fill_val, inplace=True)
                                st.success(f" Înlocuite cu mediana: {fill_val:.2f}")

                            elif strategy == 'mode':
                                fill_val = df_treated[col_to_treat].mode()[0]
                                df_treated[col_to_treat].fillna(fill_val, inplace=True)
                                st.success(f" Înlocuite cu modul: {fill_val}")

                            elif strategy == 'constant':
                                df_treated[col_to_treat].fillna(fill_value, inplace=True)
                                st.success(f"Înlocuite cu: {fill_value}")

                            elif strategy == 'interpolate':
                                df_treated[col_to_treat].interpolate(
                                    method=interp_method,
                                    limit_direction='both',
                                    inplace=True
                                )
                                st.success(f" Aplicată interpolare {interp_method}")

                            elif strategy == 'drop':
                                df_treated.dropna(subset=[col_to_treat], inplace=True)
                                n_dropped = len(df) - len(df_treated)
                                st.success(f" Eliminate {n_dropped} rânduri")

                            st.session_state['df_treated'] = df_treated

                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Înainte - Valori Lipsă", df[col_to_treat].isnull().sum())

                            with col2:
                                st.metric("După - Valori Lipsă", df_treated[col_to_treat].isnull().sum())

                            with st.expander(" Vezi Modificările"):
                                comparison_df = pd.DataFrame({
                                    'Înainte': df[col_to_treat].head(20),
                                    'După': df_treated[col_to_treat].head(20)
                                })
                                st.dataframe(comparison_df, use_container_width=True)

                            # Înlocuire cu media
                            fill_value = df['{col_to_treat}'].mean()
                            df['{col_to_treat}'].fillna(fill_value, inplace=True)

                            # Înlocuire cu modul (cel mai frecvent)
                            fill_value = df['{col_to_treat}'].mode()[0]
                            df['{col_to_treat}'].fillna(fill_value, inplace=True)

                            # Înlocuire cu valoare constantă
                            df['{col_to_treat}'].fillna({repr(fill_value)}, inplace=True)


                            # Interpolare
                            df['{col_to_treat}'].interpolate(
                                method='{interp_method}',
                                limit_direction='both',
                                inplace=True
                            )

                            # Elimină rânduri cu NaN
                            df.dropna(subset=['{col_to_treat}'], inplace=True)

                        except Exception as e:
                            st.error(f" Eroare: {str(e)}")

        else:
            st.success(" Nu există valori lipsă în dataset!")

def show_numeric_data():
    if 'df' not in st.session_state:
        st.warning("Vă rugăm să consultați secțiunea 'Încărcare, validare fișier Excel/CSV'")
        return

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("Nu există coloane numerice în dataset!")
        return

    col_for_graphs = st.selectbox("Selectează coloana pentru histogramă:", numeric_cols, key="hist_col")



    col1, col2 = st.columns([1, 1])

    with col1:

        n_bins = st.slider(
            "Număr de bin-uri:",
            min_value=10,
            max_value=100,
            value=55,
            step=1
        )

        fig = px.histogram(
            df,
            x=col_for_graphs,
            nbins=n_bins,
            title=f'Histogramă: {col_for_graphs} ({n_bins} bin-uri)',
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("### Statistici")
        stats_df = pd.DataFrame({
            'Metrică': ['Minim', 'Q1 (25%)', 'Mediană', 'Q3 (75%)', 'Maxim', 'Media', 'Std Dev'],
            'Valoare': [
                df[col_for_graphs].min(),
                df[col_for_graphs].quantile(0.25),
                df[col_for_graphs].median(),
                df[col_for_graphs].quantile(0.75),
                df[col_for_graphs].max(),
                df[col_for_graphs].mean(),
                df[col_for_graphs].std()
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

    Q1 = df[col_for_graphs].quantile(0.25)
    Q3 = df[col_for_graphs].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = df[(df[col_for_graphs] < lower_fence) | (df[col_for_graphs] > upper_fence)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df) * 100) if len(df) > 0 else 0

    fig = px.box(
        df,
        y=col_for_graphs,
        points='outliers',
        title=f'Box Plot: {col_for_graphs}'
    )
    fig.add_hline(y=lower_fence, line_dash="dash", line_color="red", annotation_text="Lower Fence")
    fig.add_hline(y=upper_fence, line_dash="dash", line_color="red", annotation_text="Upper Fence")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(" Total Valori", len(df))

    with col2:
        st.metric("Outlieri Găsiți", n_outliers)

    with col3:
        st.metric(" Procent Outlieri", f"{pct_outliers:.2f}%")

    if n_outliers > 0:
        with st.expander(" Vezi Outlierii"):
            st.dataframe(outliers[[col_for_graphs]].describe(), use_container_width=True)
            st.dataframe(outliers.head(20), use_container_width=True)

def show_categorial_data():
    if 'df' not in st.session_state:
        st.warning("Vă rugăm să consultați secțiunea 'Încărcare, validare fișier Excel/CSV'")
        return

    df = st.session_state['df'].copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        st.error("Nu există coloane text în dataset!")
        return

    col_to_filter = st.selectbox("Selectează coloana text:", cat_cols, key="filter_col")
    unique_vals = df[col_to_filter].dropna().unique().tolist()
    default_list = unique_vals[:5] if len(unique_vals) >= 5 else unique_vals
    valid_values = st.multiselect(
        "Selectează valorile valide:",
        options=unique_vals,
        default=[v for v in default_list if v in unique_vals]
    )

    st.markdown("### Definește Valori Valide")

    if valid_values:
        col1, col2 = st.columns(2)

        with col1:
            filter_method = st.radio(
                "Metodă de filtrare:",
                ['keep_only', 'mark_other'],
                format_func=lambda x: {
                    'keep_only': ' Păstrează Doar Valorile Valide',
                    'mark_other': ' Marchează Restul ca "ALTA CATEGORIE"'
                }[x]
            )

        with col2:
            case_sensitive = st.checkbox("Case sensitive", value=False)

        if st.button("  Aplică Filtrarea", type="primary"):
            df_filtered = df.copy()

            if case_sensitive:
                mask = df_filtered[col_to_filter].isin(valid_values)
            else:
                valid_lower = [v.lower() for v in valid_values]
                mask = df_filtered[col_to_filter].str.lower().isin(valid_lower)

            if filter_method == 'keep_only':
                df_filtered = df_filtered[mask]
                n_removed = len(df) - len(df_filtered)
                st.success(f"   Păstrate {len(df_filtered)} rânduri, eliminate {n_removed}")
            else:

                new_col = f'{col_to_filter}_CATEGORIE'
                df_filtered[new_col] = df_filtered[col_to_filter].where(mask, 'ALTA CATEGORIE')
                st.success(f"   Creată coloana nouă: {new_col}")

            st.session_state['df_filtered_text'] = df_filtered

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Înainte")
                st.dataframe(df[col_to_filter].value_counts().head(10), use_container_width=True)

                fig1 = px.bar(
                    x=df[col_to_filter].value_counts().head(10).index,
                    y=df[col_to_filter].value_counts().head(10).values,
                    title="Distribuție Originală"
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("### După")
                if filter_method == 'keep_only':
                    st.dataframe(df_filtered[col_to_filter].value_counts(), use_container_width=True)

                    fig2 = px.bar(
                        x=df_filtered[col_to_filter].value_counts().index,
                        y=df_filtered[col_to_filter].value_counts().values,
                        title="Distribuție Filtrată"
                    )
                else:
                    new_col = f'{col_to_filter}_CATEGORIE'
                    st.dataframe(df_filtered[new_col].value_counts(), use_container_width=True)

                    fig2 = px.bar(
                        x=df_filtered[new_col].value_counts().index,
                        y=df_filtered[new_col].value_counts().values,
                        title="Distribuție cu Categorii"
                    )

                st.plotly_chart(fig2, use_container_width=True)

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers
def show_descriptive_stats():
    import plotly.express as px
    if 'df' not in st.session_state:
        st.warning("Vă rugăm să consultați secțiunea 'Încărcare, validare fișier Excel/CSV'")
        return

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    tab_corelatie, tab_statistics = st.tabs(["Corelatie intre col. numerice", "Statistici descriptive"])

    with tab_corelatie:
        st.markdown("### Matrice de Corelație")

        corr_method = st.radio(
            "Metoda de corelație:",
            ['pearson', 'spearman', 'kendall'],
            format_func=lambda x: {
                'pearson': 'Pearson (Linear)',
                'spearman': 'Spearman (Rank)',
                'kendall': 'Kendall (Rank)'
            }[x],
            horizontal=True
        )

        corr_matrix = df[numeric_cols].corr(method=corr_method)

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title=f'Heatmap Corelație ({corr_method.capitalize()})'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Corelații Puternice (|r| > 0.7)")

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_upper = corr_matrix.where(mask)

        strong_corr = []
        for col in corr_upper.columns:
            for idx in corr_upper.index:
                val = corr_upper.loc[idx, col]
                if not pd.isna(val) and abs(val) > 0.7:
                    strong_corr.append({
                        'Variabila 1': idx,
                        'Variabila 2': col,
                        'Corelație': val,
                        'Forță': 'Foarte Puternică' if abs(val) > 0.9 else 'Puternică'
                    })

        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr).sort_values('Corelație', key=abs, ascending=False)
            st.dataframe(strong_corr_df, use_container_width=True)

            st.warning("""
                 **Multicoliniaritate Potențială!**
    
            Variabile foarte corelate pot cauza probleme în modelare:
            - Redundanță (informație duplicată)
            - Instabilitate în modele de regresie
            - Dificultăți în interpretare
    
            **Soluții:**
            - Elimină una dintre variabilele corelate
            - Folosește PCA pentru reducerea dimensionalității
            - Folosește regularizare (Ridge, Lasso)
            """)
        else:
            st.success("   Nu există corelații foarte puternice (|r| > 0.7)")
    with tab_statistics:
        if not numeric_cols:
            st.error("Nu există coloane numerice în dataset!")
            return

        col1, col2 = st.columns(2)

        with col1:
            scatter_y = st.selectbox(
                "Variabila numerică (Y):",
                numeric_cols,
                key="scatter_y"
            )

        with col2:
            scatter_x = st.selectbox(
                "Variabila numerică (X):",
                numeric_cols,
                key="scatter_x"
            )

        tab_scatter, tab_iqr = st.tabs(["Scatter Plot","Outlieri IQR"])

        with tab_scatter:
            fig = px.scatter(
                df,
                x=scatter_x,
                y=scatter_y,
                title=f"Scatter Plot: {scatter_y} vs {scatter_x}",
                opacity=0.7
            )

            fig.update_layout(
                xaxis_title=scatter_x,
                yaxis_title=scatter_y
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab_iqr:
            outliers_x = detect_outliers_iqr(df[scatter_x])
            outliers_y = detect_outliers_iqr(df[scatter_y])

            outlier_mask = outliers_x | outliers_y
            df_plot = df.copy()
            df_plot["Outlier"] = np.where(outlier_mask, "Outlier", "Normal")

            fig = px.scatter(
                df_plot,
                x=scatter_x,
                y=scatter_y,
                color="Outlier",
                title=f"Outlieri detectați prin metoda IQR",
                color_discrete_map={
                    "Normal": "#9C27B0",
                    "Outlier": "#FF5252"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            pearson_corr = df[[scatter_x, scatter_y]].corr(method="pearson").iloc[0, 1]

            st.metric(
                label="Coeficient de corelație Pearson",
                value=f"{pearson_corr:.3f}"
            )

            outlier_summary = []

            for col in numeric_cols:
                outliers = detect_outliers_iqr(df[col])
                count = outliers.sum()
                percent = (count / len(df)) * 100

                outlier_summary.append({
                    "Coloană": col,
                    "Număr Outlieri": count,
                    "Procent (%)": round(percent, 2)
                })

            outlier_df = pd.DataFrame(outlier_summary)

            st.markdown("### Outlieri detectați (metoda IQR)")
            st.dataframe(outlier_df, use_container_width=True)

def sidebar_navigation():
    st.sidebar.markdown("## Navigare:")

    sections = [
        " 1 - Incarcare si validare fisier Excel/CSV",
        " 2 - Informatii, Statistici si vizualizari descriptie",
        " 3 - Selectare coloană numerică, calcul si afisare",
        " 4 - Selectare coloană categoriala, afisare",
        " 5 - Detectarea Valorilor Anormale"
    ]

    selected = st.sidebar.radio("", sections)

    return selected

if __name__ == "__main__":
    selected_module = sidebar_navigation()

    if selected_module == " 1 - Incarcare si validare fisier Excel/CSV":
        show_data_reading()
    elif selected_module == " 2 - Informatii, Statistici si vizualizari descriptie":
        show_data_info()
    elif selected_module == " 3 - Selectare coloană numerică, calcul si afisare":
        show_numeric_data()
    elif selected_module == " 4 - Selectare coloană categoriala, afisare":
         show_categorial_data()
    elif selected_module == " 5 - Detectarea Valorilor Anormale":

         show_descriptive_stats()
