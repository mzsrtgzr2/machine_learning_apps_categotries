from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import logging
import json

from classifier import Classifier

app = Flask(__name__)

output_model_file = 'model.pcl'

@app.route('/init', methods=['POST'])
def init():
    """
    :param: data - json list of apps to train from
    :return: init the classifier
    """
    try:
        content = json.loads(request.get_data(as_text=True))
        print (content)
        clf.train(content['data'])
        return jsonify(error=None), 200
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify(error=str(e)), 500


@app.route('/getSegments', methods=['GET'])
def getSegments():
    """
    :return: Returns a list of all possible segments
    """
    try:
        return jsonify(clf.get_classes()), 200
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify(error=str(e)), 500


@app.route('/getTopSegments', methods=['POST'])
def getTopSegments ():
    """
    Maps the top n most suitable segments to the given app. The
    output will be ordered from most suitable to the least suitable
    match (by an internal scoring that needs to be defined).
    Return a list of segments with its score.
    :param: json app and n
    :return: json prediction list - segments and probability
    """
    try:
        content = json.loads(request.get_data(as_text=True))
        app = content['app']
        n = content['n']
        # make sure n is compatible with our classifier classes
        if n<1:
            raise Exception('n should be bigger than 1')
        elif n>len(clf.get_classes()):
            raise Exception('n should be less or equal to #classes')

        # predict the segment out of the description
        segments = clf.predict_one(app['description'])

        # return the top n most suitable semgnets
        return jsonify(segments[:n])
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify(error=str(e)), 500

@app.route('/getBatchSegment', methods=['POST'])
def getBatchSegment():
    """
    Maps each app to the most suitable segment. The list of apps
    are provided as input in a JSON format.
    :param app_list: json list of apps for us to predict segmets for
    :return: json list of objects with appId and predicted segment
    """
    try:
        content = json.loads(request.get_data(as_text=True))
        app_list = content['app_list']
        # predict the segments from the apps
        segments = clf.predict_batch([app['description'] for app in app_list])
        # wrap the segments in the corresponding appId and return it in json format
        return jsonify([{"appId": a['appId'], "segment": s} for a,s in zip(app_list, segments)])
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    clf = Classifier(None, output_model_file)
    CORS(app)
    app.run(port=5000)