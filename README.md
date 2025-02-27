<html>

<script type="text/javascript">

function disp()

{

var msg=document.getElementById("t1");

alert("hello "+msg.value+" exams are near you have to study");

}

function getConfirmation()

{

var ans=confirm("Do you want to continue?" );

if(ans==true)

{

alert("hello exams are near you have to study");

}

else

{

alert("hello exams are over");
}

}

function prm()

{

var fname=prompt("Hello! Whats your name");

alert("hello "+fname+" exams are near you have to study");

}

</script>

<body>

<input type=text id="t1"><br>

<input type=submit onclick="disp()" value="alert">

< input type=submit onclick="getConfirmation()" value="confirm">

<input type=submit onclick="prm()" value="prompt">

</body>

</html>







<html>

<head>

<script type="text/javascript">

function pasuser(form) {

if (form.id.value=="ty") {

if (form.pass.value=="bcs") {

alert("welcome Login successful");

} else {

alert("Invalid Password");

}
} else { alert("Invalid UserID")

}}

</script>

</head>

<body>

<center>

<form name="login">

Login Area <br>
UserID: <input type="text" name="id" ><br><br>

Password:<input type="password" name="pass"><br>

<input type="button" value="Login"onClick="pasuser(this.form)">

<input type="Reset"></form></center></body>
