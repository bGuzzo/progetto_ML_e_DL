# Anomaly Transformer: Progetto Sperimentale per l'Esame di Machine e Deep Learning

## Introduzione

Questo repository contiene il codice e gli esperimenti relativi al mio progetto sperimentale sull'Anomaly Transformer, un'architettura basata sull'attenzione per il rilevamento di anomalie nelle serie temporali. Il progetto è stato realizzato per l'esame di Machine e Deep Learning dell'A.A. 2023/2024.

## Get Started
* Installare `Python 3.6`, `PyTorch` >= 1.4.0. e CUDA.

## Contenuto del Repository

*   `model/`: Implementazione PyTorch dell'Anomaly Transformer.
*   `self_attention/`: Implementazione PyTorch di un modello con self-attention classica per Anomaly Detection
*   `solver.py`: Script per addestrare e testare il modello Anomaly Transformer.
*   `self_att_solver.py`: Script per addestrare e testare il modello con self-attention classica.
*   `grid_search.py`: (Main file) Implementazione della Grid Search per Anomaly Transformer.
*   `grid_search_self_att.py`: (Main file) Implementazione della Grid Search per il modello con self-attention classica.
*   `results/`: Cartella per salvare i risultati dell'esecuzione della grid search
*   `relazione/`: Cartella contenente la relazione (PDF) e la presentazione del progetto

## Esperimenti e Confronti

Il progetto ha incluso i seguenti esperimenti e confronti:

*   **Analisi di sensitività degli iperparametri:** Studio dell'impatto degli iperparametri sul modello, in particolare con dimensioni ridotte.
*   **Algoritmi di ottimizzazione:** Confronto delle prestazioni di diversi algoritmi di ottimizzazione (Adam, AdamW, SGD, Adadelta, RMSprop).
*   **RNN in combinazione con Anomaly Transformer:** Valutazione dell'uso di LSTM in combinazione con l'Anomaly Transformer.
*   **Anomaly Attention vs. Self Attention:** Confronto diretto tra il meccanismo di Anomaly Attention e la Self Attention classica.

## Risultati

I risultati degli esperimenti sono riportati nella relazione del progetto. In sintesi, il modello Anomaly Transformer ha dimostrato:

*   **Robustezza:** Buone prestazioni anche con dimensioni ridotte e diverse configurazioni di funzioni e parametri.
*   **Efficacia:** Superiore ad altri metodi di rilevamento anomalie nelle serie temporali.
*   **Vantaggio dell'Anomaly Attention:** Miglioramento significativo rispetto all'uso della Self Attention classica.

## Conclusioni

Il progetto ha approfondito la comprensione dell'Anomaly Transformer e ha confermato la sua efficacia nel rilevamento di anomalie nelle serie temporali. Gli esperimenti hanno evidenziato la robustezza del modello e il vantaggio dell'Anomaly Attention rispetto alla Self Attention.

## Riferimenti

*   Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly transformer: Time series anomaly detection with association discrepancy. 
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

