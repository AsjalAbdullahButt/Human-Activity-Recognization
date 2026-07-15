import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import os
import json
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set plot style for professional visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

# Create output directory for results
def create_output_dir():
    output_dir = os.path.join(os.getcwd(), 'har_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Results will be saved in: {output_dir}")
    return output_dir

# Load the datasets
def load_data():
    try:
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Preprocess data for RNN
def preprocess_data(train_data, test_data):
    # Split into features and target
    X_train = train_data.iloc[:, :-2]  # All columns except the last two (subject and activity)
    y_train = train_data.iloc[:, -1]   # Activity column
    
    X_test = test_data.iloc[:, :-2]    # All columns except the last two (subject and activity)
    y_test = test_data.iloc[:, -1]     # Activity column
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode the target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to categorical for multi-class classification
    num_classes = len(np.unique(y_train_encoded))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Reshape for RNN input [samples, time_steps, features]
    # For HAR data, we'll treat each sample as a single time step with multiple features
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    print(f"X_train shape: {X_train_reshaped.shape}")
    print(f"X_test shape: {X_test_reshaped.shape}")
    print(f"Class distribution in training set:\n{pd.Series(y_train_encoded).value_counts()}")
    
    # Save the original feature data for visualization
    return X_train_reshaped, y_train_categorical, X_test_reshaped, y_test_categorical, label_encoder, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

# Build and train RNN model
def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# Train the RNN model
def train_rnn_model(model, X_train, y_train, X_val, y_val, output_dir):
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=50,  # Increased from 15 to 50 to allow more epochs
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_rnn_model.h5'), 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,  # 100 epochs
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return model, history

# Evaluate the RNN model
def evaluate_model(model, X_test, y_test, label_encoder, output_dir, X_test_flat, y_test_encoded):
    # Evaluate the model
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    # Convert numeric predictions back to original labels
    activity_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
    
    print("\n--- RNN Model Results ---")
    report = classification_report(y_test_classes, y_pred_classes, 
                                  target_names=activity_labels, output_dict=True)
    
    # Print classification report
    print(classification_report(y_test_classes, y_pred_classes, 
                               target_names=activity_labels))
    
    # Get overall accuracy
    accuracy = report['accuracy']
    
    # Create and save metrics CSV
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Calculate metrics
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    true_neg = np.sum(cm) - (true_pos + false_pos + false_neg)
    
    # Calculate metrics for each class
    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    # Save overall metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'AUC'],
        'Value': [f"{accuracy:.2%}", f"{np.mean(sensitivity):.4f}", 
                 f"{np.mean(specificity):.4f}", f"{(np.mean(sensitivity) + np.mean(specificity))/2:.4f}"]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
    
    # VISUALIZATION 1: Enhanced Confusion Matrix
    plt.figure(figsize=(12, 10))
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Calculate confusion matrix in percentages
    cm_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot the confusion matrix with percentages
    ax = sns.heatmap(cm_percent, annot=conf_matrix, fmt='d', cmap='Blues',
                xticklabels=activity_labels, yticklabels=activity_labels,
                annot_kws={"size": 12}, vmin=0, vmax=100)
    
    # Enhance the plot
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('True', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    
    # Adjust font size of tick labels
    plt.xticks(fontsize=11, rotation=45, ha="right")
    plt.yticks(fontsize=11)
    
    # Add a color bar with percentage label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Percentage (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # VISUALIZATION 2: Normalized Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=activity_labels, yticklabels=activity_labels,
                annot_kws={"size": 12})
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('True', fontsize=14, fontweight='bold')
    plt.title('Normalized Confusion Matrix (%)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=11, rotation=45, ha="right")
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # VISUALIZATION 3: Precision, Recall, F1-Score, and Support Bar Chart
    plt.figure(figsize=(14, 8))
    
    metrics_data = {
        'Precision': [report[label]['precision'] for label in activity_labels],
        'Recall': [report[label]['recall'] for label in activity_labels],
        'F1-Score': [report[label]['f1-score'] for label in activity_labels],
    }
    
    # Set width of bars
    barWidth = 0.25
    
    # Set position of bars on X axis
    r1 = np.arange(len(activity_labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create grouped bars
    plt.bar(r1, metrics_data['Precision'], width=barWidth, label='Precision', color='#3498db', edgecolor='gray')
    plt.bar(r2, metrics_data['Recall'], width=barWidth, label='Recall', color='#2ecc71', edgecolor='gray')
    plt.bar(r3, metrics_data['F1-Score'], width=barWidth, label='F1-Score', color='#f39c12', edgecolor='gray')
    
    # Add values on top of bars
    for i, r_set in enumerate([r1, r2, r3]):
        metric_name = list(metrics_data.keys())[i]
        for j, value in enumerate(metrics_data[metric_name]):
            plt.text(r_set[j], value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Add labels and title
    plt.xlabel('Activity', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Precision, Recall, and F1-Score by Activity Class', fontsize=16, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(activity_labels))], activity_labels, rotation=45, ha='right', fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics_by_class.png'), dpi=300, bbox_inches='tight')
    
    # VISUALIZATION 4: ROC Curves for Multi-Class
    plt.figure(figsize=(12, 10))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(activity_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(activity_labels)))
    
    for i, color, label in zip(range(len(activity_labels)), colors, activity_labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{label} (AUC = {roc_auc[i]:.2f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    
    # Save predictions and metrics
    predictions_df = pd.DataFrame({
        'True_Activity': label_encoder.inverse_transform(y_test_classes),
        'Predicted_Activity': label_encoder.inverse_transform(y_pred_classes),
        'Correct': y_test_classes == y_pred_classes
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Save summary report
    with open(os.path.join(output_dir, 'har_prediction_report.txt'), 'w') as f:
        f.write(f"Human Activity Recognition - RNN Model Results\n")
        f.write(f"===========================================\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
        f.write("Performance by Activity:\n")
        for i, activity in enumerate(activity_labels):
            f.write(f"{activity}: F1-Score = {f1[i]:.4f}, Sensitivity = {sensitivity[i]:.4f}, " +
                   f"Specificity = {specificity[i]:.4f}\n")
    
    # VISUALIZATION 5: Feature visualization using PCA
    try:
        # Apply PCA to reduce dimensions to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test_flat)
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with different colors for each class
        for i, label in enumerate(activity_labels):
            mask = y_test_encoded == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, 
                       alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        
        plt.title('PCA Visualization of Sensor Data by Activity Class', fontsize=16, fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14, fontweight='bold')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14, fontweight='bold')
        plt.legend(title='Activity', fontsize=12, title_fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Could not generate PCA visualization: {e}")
    
    return y_pred_classes, accuracy

# Plot training history with enhanced visualization
def plot_training_history(history, output_dir):
    # Extract training stats
    epochs = range(1, len(history.history['accuracy']) + 1)
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # VISUALIZATION 1: Enhanced Training History
    plt.figure(figsize=(16, 7))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'o-', label='Training', linewidth=2, color='#3498db', markersize=4)
    plt.plot(epochs, val_acc, 'o-', label='Validation', linewidth=2, color='#e74c3c', markersize=4)
    
    # Find best validation accuracy
    best_val_epoch = np.argmax(val_acc) + 1
    best_val_acc = val_acc[best_val_epoch - 1]
    plt.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.7, 
               label=f'Best val_acc: {best_val_acc:.4f} (epoch {best_val_epoch})')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Model Accuracy over Epochs', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'o-', label='Training', linewidth=2, color='#3498db', markersize=4)
    plt.plot(epochs, val_loss, 'o-', label='Validation', linewidth=2, color='#e74c3c', markersize=4)
    
    # Find best (lowest) validation loss
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_val_loss_epoch - 1]
    plt.axvline(x=best_val_loss_epoch, color='green', linestyle='--', alpha=0.7,
               label=f'Best val_loss: {best_val_loss:.4f} (epoch {best_val_loss_epoch})')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Model Loss over Epochs', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    
    # VISUALIZATION 2: Learning Curve (single plot with dual y-axis)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    color1 = '#3498db'
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', color=color1, fontsize=14, fontweight='bold')
    line1 = ax1.plot(epochs, train_acc, 'o-', label='Train Accuracy', color=color1, linewidth=2.5, markersize=4)
    line2 = ax1.plot(epochs, val_acc, 's-', label='Validation Accuracy', color='#9b59b6', linewidth=2.5, markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([min(min(train_acc), min(val_acc))-0.05, 1.02])
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color2 = '#e74c3c'
    ax2.set_ylabel('Loss', color=color2, fontsize=14, fontweight='bold')
    line3 = ax2.plot(epochs, train_loss, '^-', label='Train Loss', color=color2, linewidth=2.5, markersize=4)
    line4 = ax2.plot(epochs, val_loss, 'v-', label='Validation Loss', color='#f39c12', linewidth=2.5, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Calculate good y-limits for the loss axis
    loss_min = min(min(train_loss), min(val_loss))
    loss_max = max(max(train_loss[1:]), max(val_loss[1:]))  # Skip first epoch if very large loss
    ax2.set_ylim([max(0, loss_min-0.1), loss_max*1.1])
    
    # Add vertical line at best validation accuracy
    ax1.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.5)
    ax1.text(best_val_epoch+0.1, 0.5, f'Best Model (epoch {best_val_epoch})', 
             transform=ax1.get_xaxis_transform(), ha='left', va='center', fontsize=10, color='green')
    
    # Combine legends from both axes
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=12)
    
    plt.title('Training and Validation Learning Curves', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    
    # Save history data as CSV
    hist_df = pd.DataFrame({
        'epoch': epochs,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    })
    hist_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

# Main function
def main():
    # Create output directory
    output_dir = create_output_dir()
    
    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # Preprocess data
    X_train, y_train, X_test, y_test, label_encoder, X_train_flat, X_test_flat, y_train_encoded, y_test_encoded = preprocess_data(train_data, test_data)
    
    # Split validation set
    val_split = 0.2
    val_samples = int(len(X_train) * val_split)
    X_val = X_train[-val_samples:]
    y_val = y_train[-val_samples:]
    X_train = X_train[:-val_samples]
    y_train = y_train[:-val_samples]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    
    # VISUALIZATION: Class distribution visualization
    plt.figure(figsize=(12, 6))
    class_names = label_encoder.classes_
    class_counts = pd.Series(y_train_encoded).value_counts().sort_index()
    
    # Create horizontal bar chart
    ax = sns.barplot(x=class_counts.values, y=class_names, palette='Blues_d', orient='h')
    
    # Add count labels to the bars
    for i, v in enumerate(class_counts.values):
        ax.text(v + 20, i, str(v), va='center', fontweight='bold')
    
    plt.title('Class Distribution in Training Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Samples', fontsize=14, fontweight='bold')
    plt.ylabel('Activity Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    rnn_model = build_rnn_model(input_shape, num_classes)
    
    # Try to visualize model architecture
    try:
        tf.keras.utils.plot_model(
            rnn_model, 
            to_file=os.path.join(output_dir, 'model_architecture.png'),
            show_shapes=True, 
            show_layer_names=True,
            dpi=200,
            expand_nested=True
        )
    except Exception as e:
        print(f"Could not generate model architecture visualization: {e}")
        print("To fix this, install: pip install pydot graphviz")
        print("And download Graphviz from: https://graphviz.gitlab.io/download/")
    
    # Train model
    trained_model, history = train_rnn_model(rnn_model, X_train, y_train, X_val, y_val, output_dir)
    
    # Plot enhanced training history
    plot_training_history(history, output_dir)
    
    # Evaluate model with enhanced visualizations
    y_pred, accuracy = evaluate_model(trained_model, X_test, y_test, label_encoder, output_dir, X_test_flat, y_test_encoded)
    
    # Save model
    trained_model.save(os.path.join(output_dir, 'har_rnn_model.h5'))
    print(f"\nRNN model saved successfully! Final test accuracy: {accuracy:.4f}")
    
    # Create a performance dashboard summary
    plt.figure(figsize=(16, 12))
    plt.suptitle('Human Activity Recognition - RNN Model Performance Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Get class-specific metrics
    report = classification_report(np.argmax(y_test, axis=1), y_pred, 
                                  target_names=label_encoder.classes_, output_dict=True)
    
    # Activity performance bar chart
    activities = list(label_encoder.classes_)
    f1_scores = [report[act]['f1-score'] for act in activities]
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(activities, f1_scores, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(activities))))
    plt.title('F1-Score by Activity', fontsize=14, fontweight='bold')
    plt.xlabel('Activity', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Key metrics table
    plt.subplot(2, 2, 2)
    plt.axis('off')
    metrics_table = {
        'Metric': ['Accuracy', 'Precision (avg)', 'Recall (avg)', 'F1-Score (avg)'],
        'Value': [
            f"{accuracy:.2%}",
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{report['macro avg']['f1-score']:.4f}"
        ]
    }
    
    plt.table(
        cellText=[[metrics_table['Metric'][i], metrics_table['Value'][i]] for i in range(len(metrics_table['Metric']))],
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center',
        colColours=['#D6EAF8', '#D6EAF8'],
        colWidths=[0.5, 0.3],
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    plt.title('Key Performance Metrics', fontsize=14, fontweight='bold')
    
    # Mini confusion matrix in quadrant 3
    plt.subplot(2, 2, 3)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[a[:3] for a in activities],  # Shortened labels for space
                yticklabels=[a[:3] for a in activities],
                cbar=False)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    
    # Training history in quadrant 4
    plt.subplot(2, 2, 4)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, history.history['accuracy'], 'o-', label='Train Acc', color='#3498db', linewidth=2)
    plt.plot(epochs, history.history['val_accuracy'], 'o-', label='Val Acc', color='#e74c3c', linewidth=2)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
