import os
import io
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def dbscan_clustering(X):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN clustering
    eps = 0.4  # distance epsilon
    min_samples = 4
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    return clusters

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/second.html')
def second():
    return render_template('second.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/table.html')
def table():
    return render_template('table.html')

@app.route('/addresult.html')
def add():
    return render_template('addresult.html')


@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    if request.method == 'POST':
        # Get the uploaded dataset from the request
        uploaded_dataset = request.files.get('dataset')  

        # Read the uploaded dataset
        if uploaded_dataset:
            df = pd.read_csv(uploaded_dataset)
        else:
            df = None

        # Perform clustering and prepare the response
        response = predict()

        # Render the template with the response dictionary included in the context
        return render_template('segmentation.html', response=response)
        print("response send")
    else:
        # Render segmentation.html without dataset if it's a GET request
        return render_template('segmentation.html', response=None)
        print("response not sent")

@app.route('/conclusion.html')
def conclusion():
    return render_template('conclusion.html')

@app.route('/predict', methods=['POST'])

def predict():
    try:    
        # Load the data
        data = request.files['files']
        X = pd.read_csv(data)

        # Extract file name and column names
        filename = data.filename
        column_names = X.columns.tolist()

        # Perform DBSCAN clustering
        clusters = dbscan_clustering(X[['Annual Income (k$)', 'Spending Score (1-100)']])

        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Get cluster details
        cluster_details = {}
        unique_clusters = np.unique(clusters)
        for cluster_label in unique_clusters:
            if cluster_label != -1:  # Ignore noise points
                cluster_points = X[clusters == cluster_label]
                cluster_center = cluster_points.mean(axis=0)
                cluster_size = len(cluster_points)
                cluster_details[str(cluster_label)] = {  # Convert cluster_label to string
                    'center': cluster_center.tolist(),
                    'size': cluster_size
                }

        # Print cluster details
        print("Cluster Details:")
        for label, details in cluster_details.items():
            cluster_name = f"Cluster {label}"
            num_points = details['size']
            mean_range = f"{details['center'][0]:.2f} (Annual Income), {details['center'][1]:.2f} (Spending Score)"
            print(f"{cluster_name}: Number of Points - {num_points}, Mean Range - {mean_range}")

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=X, hue=clusters, palette='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend(title='Cluster')

        # Save the plot in the static folder
        img_path = os.path.join(app.static_folder, 'cluster_plots.png')
        plt.savefig(img_path)

        plt.close()

        # Prepare dataset object to pass to frontend
        dataset = {
            'filename': filename,
            'column_names': column_names,
            'values': X.values.tolist()  # Convert dataframe to list of lists
        }

        response = {
            'img_path': '/static/cluster_plots.png',  # Return the path relative to the static folder
            'dataset': dataset,  # Pass the dataset object
            'num_clusters': num_clusters,
            'cluster_details': cluster_details
        }
        print("Number of Clusters := ",num_clusters)
        print("Cluster Details \t",cluster_details)
        return jsonify(response)
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
