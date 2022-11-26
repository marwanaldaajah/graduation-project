from flask import Flask
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
api = Api(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class subject(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(100), nullable=False)
	
	

	def __repr__(self):
		return f"subject(name = {name})"

subject_put_args = reqparse.RequestParser()
subject_put_args.add_argument("name", type=str, help="Name of the subject is required", required=True)

subject_update_args = reqparse.RequestParser()
video_update_args.add_argument("name", type=str, help="Name of the video is required")

resource_fields = {
	'id': fields.Integer,
	'name': fields.String,
}

class subject(Resource):
	@marshal_with(resource_fields)
	def get(self, subject_id):
		result = subjectModel.query.filter_by(id=subject_id).first()
		if not result:
			abort(404, message="Could not find subject with that id")
		return result

	@marshal_with(resource_fields)
	def put(self, subject_id):
		args = subject_put_args.parse_args()
		result = subjectModel.query.filter_by(id=subject_id).first()
		if result:
			abort(409, message="subject id taken...")

		subject = subjectModel(id=video_id, name=args['name'])
		db.session.add(subject)
		db.session.commit()
		return subject, 201

	@marshal_with(resource_fields)
	def patch(self, subject_id):
		args = subject_update_args.parse_args()
		result = subjectModel.query.filter_by(id=subject_id).first()
		if not result:
			abort(404, message="subject doesn't exist, cannot update")

		if args['name']:
			result.name = args['name']

		db.session.commit()

		return result


	def delete(self, subject_id):
		abort_if_subject_id_doesnt_exist(video_id)
		del subject[subject_id]
		return '', 204


api.add_resource(subject, "/subject/<int:subject_id>")

if __name__ == "__main__":
	app.run(debug=True)