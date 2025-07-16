import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'
from graphviz import Digraph





# --- 1. Daten vorbereiten ---
df = pd.read_csv("pulsar_data.csv", sep=',', quotechar='\"', on_bad_lines='warn', low_memory=False, parse_dates=False)
df = df.dropna()

X = df.drop('target_class', axis=1)
y = df['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

# --- SVM Modelle trainieren und Ergebnisse speichern ---
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
models_svm = {}
y_pred_svm = {}
conf_matrices_svm = {}

for kernel in kernels:
    model_svm = SVC(kernel=kernel, C=4)
    model_svm.fit(X_train_scaled, y_train)
    models_svm[kernel] = model_svm
    y_pred_svm[kernel] = model_svm.predict(X_test_scaled)
    conf_matrices_svm[kernel] = confusion_matrix(y_test, y_pred_svm[kernel])

# --- Daten für Plotly im PCA-Raum vorbereiten (für SVM) ---
plot_df = pd.DataFrame({
    'PCA1': X_test_pca[:, 0],
    'PCA2': X_test_pca[:, 1],
    'True Label': y_test.values,
    'Predicted Label': y_pred_svm['linear']
})

# --- NN Daten vorbereiten (Tensor) ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- 2. Modell-Definitionen ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

class ComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# --- 3. Trainingsfunktion ---
def train_model(model, name, epochs=20):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    train_losses = []
    
    for epoch in range(epochs):
        current_epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_epoch_loss += loss.item()
            
        avg_epoch_loss = current_epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
    return model, train_losses

# --- 4. Evaluationsfunktion ---
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor)
        y_pred_class = (y_pred_proba > 0.5).float()
        
        y_true_np = y_test_tensor.numpy()
        y_pred_class_np = y_pred_class.numpy()

        cm = confusion_matrix(y_true_np, y_pred_class_np)
        
        accuracy = accuracy_score(y_true_np, y_pred_class_np)
        precision = precision_score(y_true_np, y_pred_class_np, pos_label=1, zero_division=0)
        recall = recall_score(y_true_np, y_pred_class_np, pos_label=1, zero_division=0)
        f1 = f1_score(y_true_np, y_pred_class_np, pos_label=1, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return cm, metrics

# --- 5. Beide Modelle trainieren und evaluieren ---

simple_model, simple_train_losses = train_model(SimpleNN(X_train_scaled.shape[1]), "SimpleNN")

complex_model, complex_train_losses = train_model(ComplexNN(X_train_scaled.shape[1]), "ComplexNN")

# Ergebnisse der NN-Modelle speichern
nn_results = {}
for model_name, model_obj in [("SimpleNN", simple_model), ("ComplexNN", complex_model)]:
    cm, metrics = evaluate_model(model_obj, X_test_tensor, y_test_tensor)
    nn_results[model_name] = {'confusion_matrix': cm, 'metrics': metrics}
    
def draw_model_architecture():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR')  # left to right

    dot.node('input', 'Input')
    dot.node('fc1', 'Linear\n(8 → 16)')
    dot.node('relu1', 'ReLU')
    dot.node('fc2', 'Linear\n(16 → 8)')
    dot.node('relu2', 'ReLU')
    dot.node('fc3', 'Linear\n(8 → 1)')
    dot.node('sigmoid', 'Sigmoid')
    dot.node('output', 'Output')

    dot.edges([('input', 'fc1'), ('fc1', 'relu1'), ('relu1', 'fc2'),
               ('fc2', 'relu2'), ('relu2', 'fc3'), ('fc3', 'sigmoid'), ('sigmoid', 'output')])

    dot.render('assets/model_topology', view=False)

def draw_complex_model_architecture():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR')

    dot.node('input', 'Input')
    dot.node('fc1', 'Linear\n(8 → 64)')
    dot.node('relu1', 'ReLU')
    dot.node('dropout1', 'Dropout(0.3)')
    dot.node('fc2', 'Linear\n(64 → 32)')
    dot.node('relu2', 'ReLU')
    dot.node('dropout2', 'Dropout(0.3)')
    dot.node('fc3', 'Linear\n(32 → 16)')
    dot.node('relu3', 'ReLU')
    dot.node('fc4', 'Linear\n(16 → 1)')
    dot.node('sigmoid', 'Sigmoid')
    dot.node('output', 'Output')

    dot.edges([
        ('input', 'fc1'), ('fc1', 'relu1'), ('relu1', 'dropout1'),
        ('dropout1', 'fc2'), ('fc2', 'relu2'), ('relu2', 'dropout2'),
        ('dropout2', 'fc3'), ('fc3', 'relu3'), ('relu3', 'fc4'),
        ('fc4', 'sigmoid'), ('sigmoid', 'output')
    ])

    dot.render('assets/model_topology_complex', view=False)

draw_model_architecture()
draw_complex_model_architecture()

def create_weight_heatmaps(model, title_prefix):
    figures = []

    # Rekursives Durchlaufen aller Layer
    def get_all_linear_layers(m):
        layers = []
        for layer in m.children():
            if isinstance(layer, nn.Linear):
                layers.append(layer)
            elif list(layer.children()):
                layers.extend(get_all_linear_layers(layer))
        return layers

    linear_layers = get_all_linear_layers(model)

    for i, layer in enumerate(linear_layers):
        weights = layer.weight.detach().numpy()
        fig = go.Figure(data=go.Heatmap(
            z=weights,
            colorscale='Viridis',
            colorbar=dict(title='Gewicht'),
            zmid=0
        ))
        fig.update_layout(
            title=f'{title_prefix} – Layer {i+1}: Gewichtsmatrix ({weights.shape[0]}×{weights.shape[1]})',
            xaxis_title='Eingänge',
            yaxis_title='Neuronen',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        figures.append(fig)
    
    return figures

simple_weight_figs = create_weight_heatmaps(simple_model, "SimpleNN")
complex_weight_figs = create_weight_heatmaps(complex_model, "ComplexNN")


# --- Dash App Initialisierung ---
app = dash.Dash(__name__)
app.title = "Pulsar Klassifikation"

# --- Funktionen zur Erstellung der Plotly Figuren ---
def create_confusion_matrix_figure(cm, title):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['0: Kein Pulsar', '1: Pulsar'],
        y=['0: Kein Pulsar', '1: Pulsar'],
        colorscale='Blues',
        showscale=True,
        zmin=0,
        text=cm,
        texttemplate="%{text}",
        textfont={"size":16}
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title='Vorhergesagte Klasse', tickangle=-45),
        yaxis=dict(title='Tatsächliche Klasse', autorange='reversed'),
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        width=550 # Breite für Konfusionsmatrix erhöht
    )
    return fig

def create_nn_metrics_line_chart(nn_results):
    fig = go.Figure()
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    simple_metrics = nn_results['SimpleNN']['metrics']
    fig.add_trace(go.Scatter(
        x=metrics_names,
        y=[simple_metrics['accuracy'], simple_metrics['precision'], simple_metrics['recall'], simple_metrics['f1_score']],
        mode='lines+markers',
        name='SimpleNN',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    complex_metrics = nn_results['ComplexNN']['metrics']
    fig.add_trace(go.Scatter(
        x=metrics_names,
        y=[complex_metrics['accuracy'], complex_metrics['precision'], complex_metrics['recall'], complex_metrics['f1_score']],
        mode='lines+markers',
        name='ComplexNN',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Vergleich der Performance-Metriken (Testdaten)',
        xaxis_title='Metrik',
        yaxis_title='Wert',
        yaxis_range=[0, 1],
        legend_title='Modell',
        hovermode='x unified',
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig



# --- Dash Layout ---
app.layout = html.Div([
    html.H1("Pulsar Klassifikation mit SVM und Neuronalen Netzen", style={'textAlign': 'center', 'color': '#333'}),

    html.Hr(),

    html.H2("Aufgabe 2: SVM Performance Vergleich", style={'textAlign': 'center', 'color': '#555'}),
    html.Div([
        dcc.Dropdown(
            id='svm-kernel-dropdown',
            options=[{'label': k.capitalize(), 'value': k} for k in kernels],
            value='linear',
            style={'width': '50%', 'margin': '20px auto', 'font-size': '1.1em'}
        ),
    ], style={'display': 'flex', 'justify-content': 'center'}),

    dcc.Graph(id='svm-graph', style={'width': '80%', 'margin': '0 auto'}),
    dcc.Graph(id='svm-confusion-matrix-table', style={'width': '600px', 'margin': '0 auto', 'padding-top': '20px'}), # Feste Breite und zentriert

    html.Hr(),

    html.H2("Aufgabe 3: Neuronale Netze Performance Vergleich", style={'textAlign': 'center', 'color': '#555'}),
    
    dcc.Graph(
        id='simple-nn-confusion-matrix',
        figure=create_confusion_matrix_figure(nn_results['SimpleNN']['confusion_matrix'], 'SimpleNN Konfusionsmatrix'),
        style={'width': '600px', 'margin': '0 auto', 'padding-top': '20px'} # Feste Breite und zentriert
    ),
    dcc.Graph(
        id='complex-nn-confusion-matrix',
        figure=create_confusion_matrix_figure(nn_results['ComplexNN']['confusion_matrix'], 'ComplexNN Konfusionsmatrix'),
        style={'width': '600px', 'margin': '0 auto', 'padding-top': '20px'} # Feste Breite und zentriert
    ),

    html.H3("Vergleich der Metriken (Accuracy, Precision, Recall, F1-Score)", style={'textAlign': 'center', 'margin-top': '30px'}),
    dcc.Graph(
        id='nn-metrics-comparison-chart',
        figure=create_nn_metrics_line_chart(nn_results),
        style={'width': '80%', 'margin': '0 auto', 'padding-top': '20px'}
    ),

    html.Hr(),

    html.H2("Aufgabe 4: Visualisierung Neuronale Netze", style={'textAlign': 'center', 'color': '#555'}),

    dcc.Graph(
        id='nn-training-loss-chart',
        figure=go.Figure(
            data=[
                go.Scatter(y=simple_train_losses, mode='lines', name='SimpleNN Training Loss'),
                go.Scatter(y=complex_train_losses, mode='lines', name='ComplexNN Training Loss')
            ],
            layout=go.Layout(
                title='Training Loss Verlauf über Epochen',
                xaxis_title='Epoche',
                yaxis_title='Durchschnittlicher Loss',
                hovermode='x unified',
                height=500,
                margin=dict(l=40, r=40, t=60, b=40)
            )
        ),
        style={'width': '80%', 'margin': '0 auto', 'padding-top': '20px'}
    ),
    html.Img(src='/assets/model_topology.png',
         style={'width': '100%', 'display': 'block', 'margin': '30px auto'}),

    html.Img(src='/assets/model_topology_complex.png',
         style={'width': '100%', 'display': 'block', 'margin': '30px auto'}),
    html.H3("SimpleNN Gewichtsmatrizen", style={'textAlign': 'center'}),
*[
    dcc.Graph(figure=fig, style={'width': '600px', 'margin': '20px auto'})
    for fig in simple_weight_figs
],
    html.H3("ComplexNN Gewichtsmatrizen", style={'textAlign': 'center'}),
*[
    dcc.Graph(figure=fig, style={'width': '600px', 'margin': '20px auto'})
    for fig in complex_weight_figs
],
    html.P("In diesem Falle ist das einfachere Netz völlig ausreichend, da wir nur 8 Eingabeparameter haben, aus denen eine ja/nein Entscheidung gefällt werden soll. Normalerweise würde man neuronale Netze eher nehmen um beispielsweise handgemalte Zahlen zu erkennen. Dann hätte man so viele Eingabeparameter, wie pixel und 10 verschiedene mögliche Ausgaben.")
])

# --- Dash Callbacks (für SVM, wie gehabt) ---
@app.callback(
    Output('svm-graph', 'figure'),
    Output('svm-confusion-matrix-table', 'figure'),
    [Input('svm-kernel-dropdown', 'value')]
)
def update_svm_plots(selected_kernel):
    model = models_svm[selected_kernel]

    h = .08
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = pca.inverse_transform(grid_pca)

    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    contour = go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdBu',
        opacity=0.4,
        showscale=False,
        name='Entscheidungsgrenze'
    )

    scatter = go.Scatter(
        x=X_test_pca[:, 0],
        y=X_test_pca[:, 1],
        mode='markers',
        marker=dict(
            color=y_test,
            colorscale='Viridis',
            showscale=False,
            size=6,
            line=dict(width=0.5, color='black')
        ),
        name='Testdaten (Tatsächlich)'
    )

    layout = go.Layout(
        title=f'SVM mit Kernel: {selected_kernel} - Entscheidungsgrenze im PCA-Raum',
        xaxis=dict(title='PCA Komponente 1'),
        yaxis=dict(title='PCA Komponente 2'),
        height=700,
        hovermode='closest'
    )

    cm = conf_matrices_svm[selected_kernel]
    heatmap_fig = create_confusion_matrix_figure(cm, f'SVM Konfusionsmatrix – Kernel: {selected_kernel}')

    return go.Figure(data=[contour, scatter], layout=layout), heatmap_fig


if __name__ == '__main__':
    app.run(debug=True)