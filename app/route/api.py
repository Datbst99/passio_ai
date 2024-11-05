from flask import Blueprint, jsonify, request
from ..controller.ApiController import ApiController
api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/text-speech", methods=["POST"])
def text_speech():
    return ApiController.text_to_speech()

@api_blueprint.route("/upload", methods=["POST"])
def upload():
    return ApiController.upload_audio()