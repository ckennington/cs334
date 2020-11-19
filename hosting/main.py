import sys
import os
import shutil

from flask import Flask, request, jsonify
import numpy
import pickle
import pandas as pd
#import joblib

app = Flask(__name__)

# inputs
model_directory = 'model'
model_file_name = '%s/model_v1.pkl' % model_directory

@app.route('/evaluate', methods=['POST'])
def evaluate():
      try:
          loaded_model = pickle.load(open('model_v1.pkl', 'rb'))
          sample = request.json
          print("{}".format(sample))
          result = loaded_model.predict(numpy.array(sample).reshape(1,-1))
          
          prediction = "{}".format(int(result))
          return jsonify({"prediction": prediction})

      except Exception as e:
          return jsonify({'error': str(e), 'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        loaded_model = pickle.load(open('model_v1.pkl', 'rb'))
        print('model loaded')

    except Exception as e:
        print('No model here')

    app.run(host='0.0.0.0', port=port, debug=True)

