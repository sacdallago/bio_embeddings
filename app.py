import re
from extractor import extract_embedding
from flask import Flask, request, abort
from flask_cors import cross_origin

app = Flask(__name__)


@app.route('/')
@cross_origin()
def get_location():
    query = request.args.get('q')

    if query:
        if UNIPROT_AC_REGEX.match(query):
            results = extract_locations_from_URL("https://www.uniprot.org/uniprot/{}.xml".format(query))
            return results.to_json(orient="records")
        else:
            abort(400)

    else:
        abort(404)


if __name__ == '__main__':
    app.run()
