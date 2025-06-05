from flask import Blueprint, jsonify

bp = Blueprint('eyes', __name__, url_prefix='/eyes')

@bp.route('/ping')
def ping():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

__all__ = ['bp']
