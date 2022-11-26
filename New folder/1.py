from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello():
    return "hello "
users={"rana":"rana","marwan":"marwan","loay":"loay","mohammad":"mohammad"}

@app.route('/user')
def show_user_overview():
    users_str="<br>".join(users.values())
    return'<h1>Our users</h1><br>{}'.format(users_str)



if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")    
    