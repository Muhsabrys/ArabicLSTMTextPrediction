# Arabic Text Prediction using LSTM

## Overview
This repository contains code for training and testing a lightweight LSTM model for Arabic text prediction. The model utilizes the Long Short-Term Memory (LSTM) architecture to predict the next word in a sequence of Arabic text.

### Project Workflow
The project involves the following steps:
1. **Reading Arabic Text Data:** The text data is loaded from `arabic_texts.txt`.
2. **Tokenization:** The text data is tokenized into sequences of integers using the Keras Tokenizer.
3. **Creating Input and Target Sequences:** Input sequences and corresponding target words are generated.
4. **Padding Sequences:** Input sequences are padded to ensure uniform length.
5. **Building the LSTM Model:** A lightweight LSTM model is implemented using Keras.
6. **Training the Model:** The model is trained on the prepared input and target sequences.
7. **Saving the Model and Tokenizer:** The trained model and tokenizer are saved for later use.
8. **Testing the Model:** The saved model is loaded, and predictions are generated for new input text.

---

## Dependencies
Ensure you have the following installed:
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pickle

---

## File Structure
```
├── arabic_texts.txt          # Arabic sentences (one per line)
├── train_lstm.py             # Script for training the LSTM model
├── test_lstm.py              # Script for testing the trained model
├── tokenizer.pickle          # Saved tokenizer
├── light_lstm_model.h5       # Saved LSTM model
```

---

## Usage

### Training the Model
1. Prepare your Arabic text data in `arabic_texts.txt`.
2. Run the following command to train the model:
   ```bash
   python train_lstm.py
   ```
   This will save `tokenizer.pickle` and `light_lstm_model.h5`.

### Testing the Model
1. Ensure `tokenizer.pickle` and `light_lstm_model.h5` exist in the directory.
2. Run the script to test the model on new input text:
   ```bash
   python test_lstm.py
   ```

---

## Example Usage

### Input:
```python
input_text = "مساء"
```
### Expected Output:
```
Input text: مساء
Predicted word: الخير
```

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

---

## License
This project is licensed under the MIT License.
