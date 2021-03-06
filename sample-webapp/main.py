from datetime import datetime
import logging
import os

from flask import Flask, redirect, render_template, request

from google.cloud import datastore
from google.cloud import storage
from google.cloud import vision


CLOUD_STORAGE_BUCKET = os.environ.get('CLOUD_STORAGE_BUCKET')


app = Flask(__name__)


@app.route('/')
def homepage():
    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Use the Cloud Datastore client to fetch information from Datastore about
    # each photo.
    query = datastore_client.query(kind='Faces')
    image_entities = list(query.fetch())

    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template('homepage.html', image_entities=image_entities)


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    photo = request.files['file']

    # Create a Cloud Storage client.
    storage_client = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(photo.filename)
    blob.upload_from_string(
            photo.read(), content_type=photo.content_type)

    # Make the blob publicly viewable.
    blob.make_public()

    vision_client = vision.ImageAnnotatorClient()

    source_uri = 'gs://{}/{}'.format(CLOUD_STORAGE_BUCKET, blob.name)

    image = vision.types.Image(
        source=vision.types.ImageSource(gcs_image_uri=source_uri))

    faces = vision_client.face_detection(image).face_annotations

    likelihoods = [
            'Unknown', 'Very Unlikely', 'Unlikely', 'Possible', 'Likely',
            'Very Likely']

    happyFace=0
    angerFace=0
    surpriseFace=0

    for face in faces:
        face_joy = likelihoods[face.joy_likelihood]
        face_anger = likelihoods[face.anger_likelihood]
        face_surprise = likelihoods[face.surprise_likelihood]
        if face_anger=='Likely' or face_anger=='Very Likely':
            angerFace+=1
        if face_joy=='Likely' or face_joy=='Very Likely':
            happyFace+=1
        if face_surprise=='Likely' or face_surprise=='Very Likely':
            surpriseFace+=1

    face_joy=str(happyFace)
    face_anger=str(angerFace)
    face_surprise=str(surpriseFace)
    datastore_client = datastore.Client()
    current_datetime = datetime.now()
    kind = 'Faces'
    name = blob.name
    key = datastore_client.key(kind, name)
    entity = datastore.Entity(key)
    entity['blob_name'] = blob.name
    entity['image_public_url'] = blob.public_url
    entity['timestamp'] = current_datetime
    entity['joy'] = face_joy
    entity['surprise']= face_surprise
    entity['anger']= face_anger
    datastore_client.put(entity)
    return redirect('/')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
