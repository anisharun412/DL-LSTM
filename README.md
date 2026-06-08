# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement

Named Entity Recognition (NER) is an important Natural Language Processing (NLP) task that involves identifying and classifying named entities present in text into predefined categories such as persons, organizations, locations, dates, events, and geopolitical entities. The objective of this experiment is to build a Bidirectional Long Short-Term Memory (BiLSTM) model capable of automatically identifying and classifying named entities from textual sentences.

### Dataset

The dataset used is ner_dataset.csv, which contains words and their corresponding named entity tags.

Entity categories present in the dataset:

geo – Geographical Entity
org – Organization
per – Person
gpe – Geopolitical Entity
tim – Time Indicator
art – Artifact
eve – Event
nat – Natural Phenomenon

The dataset is preprocessed by:

Filling missing values
Grouping words into sentences
Converting words and tags into numerical indices
Padding sequences to a fixed length
Splitting the dataset into training and testing sets

## DESIGN STEPS
STEP 1:

Import the required Python libraries and load the NER dataset.

STEP 2:

Perform preprocessing by handling missing values and extracting unique words and entity tags.

STEP 3:

Convert words and tags into numerical indices and group words into sentences.

STEP 4:

Pad all sequences to a fixed length and split the dataset into training and testing sets.

STEP 5:

Create DataLoaders and define the BiLSTM architecture consisting of an embedding layer, bidirectional LSTM layer, and fully connected layer.

STEP 6:

Define the loss function and optimizer, then train the model using the training dataset.

STEP 7:

Evaluate the trained model using the testing dataset and generate performance metrics.

STEP 8:

Perform inference on sample sentences and display predicted named entity tags.

## PROGRAM

### Name: Arunsamy D

### Register Number: 212224240016

```python
class BiLSTMTagger(nn.Module):

    def __init__(
            self,
            vocab_size,
            tagset_size,
            embedding_dim=128,
            hidden_dim=128):

        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding( vocab_size, embedding_dim )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear( hidden_dim * 2, tagset_size )

    def forward(self, input_ids):

        embeddings = self.embedding( input_ids )

        lstm_out, _ = self.bilstm( embeddings )

        logits = self.fc( lstm_out )

        return logits

# model, loss, optimizer
model = BiLSTMTagger(
    vocab_size=len(word2idx) + 1,
    tagset_size=len(tag2idx)
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)

            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model( input_ids )

            loss = loss_fn( 
                outputs.view(-1, outputs.shape[-1]), 
                labels.view(-1) 
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = ( total_train_loss / len(train_loader) )

        train_losses.append( avg_train_loss )

        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)

                labels = batch["labels"].to(device)

                outputs = model( input_ids )

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]),
                    labels.view(-1)
                )

                total_val_loss += loss.item()

        avg_val_loss = ( total_val_loss / len(test_loader) )

        val_losses.append( avg_val_loss )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses

```

## OUTPUT

## Dataset Information
<img width="855" height="826" alt="image" src="https://github.com/user-attachments/assets/faf6569a-4f07-4acd-8f5c-7b82b3735177" />

### Unique Words and Entity Tags
<img width="592" height="302" alt="image" src="https://github.com/user-attachments/assets/363b6202-04b0-409a-9c14-c21a57628d6e" />

### Sentence Length Distribution
<img width="560" height="413" alt="download" src="https://github.com/user-attachments/assets/46025422-d798-418a-8ab6-001326da077b" />

### Training, Validation Loss, Loss Vs Epoch Plot
<img width="759" height="530" alt="image" src="https://github.com/user-attachments/assets/75055198-a106-453f-aba1-3ad6cb09b1c1" />
<br>
<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/548ba4cd-1a17-4c9a-8288-a66ce08ba9b6" />

### Accuracy
<img width="677" height="250" alt="image" src="https://github.com/user-attachments/assets/b908588a-4752-4370-9192-0f395d4e0c95" />

### Sample Text Prediction
<img width="548" height="499" alt="image" src="https://github.com/user-attachments/assets/2728fc85-58e5-4fe9-9b5e-4dff8e664ecc" />


## RESULT
Thus, a Bidirectional Long Short-Term Memory (BiLSTM) based Named Entity Recognition (NER) model was successfully developed and trained using the NER dataset. The model was able to identify and classify named entities such as persons, organizations, locations, geopolitical entities, and events from textual data with good accuracy.
