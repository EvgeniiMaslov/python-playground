## Music genre classification



Using a machine learning model it is possible to automatically classify music genres {"classical", "jazz", "country", "pop", "rock", "metal"}. 



**You need to install following libraries: **

* [Flask](#https://flask.palletsprojects.com/)
* [werkzeug](https://werkzeug.palletsprojects.com/)
* [pickle](#https://docs.python.org/3/library/pickle.html)
* [xgboost](#https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [numpy](#https://numpy.org/)
* [sklearn](#https://scikit-learn.org/stable/)
* [python_speach_features](#https://python-speech-features.readthedocs.io/en/latest/)
* [scipy](#https://www.scipy.org/)



**How to use:**

Past following in terminal:

```
python main.py
```

Then open link from console in your browser, select file you want to classify and click `predict` button.

**Note:** currently supports only **.wav** files 

<hr>

**If you want to train model:**

1. Download the data by:

   ```
   python load_dataset.py
   ```

   or make `data` folder and extract your own data there.

2. Train the model by:

   ```
   python train.py --method
   ```

   Viable two methods for preprocessing .wav files:

   * mfcc - [mel-frequency cepstrum](#https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), provides better model classification accuracy
   * fft - [fast fourier transform](#https://en.wikipedia.org/wiki/Fast_Fourier_transform)

3. After training your model is ready to use.



