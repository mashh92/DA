
<html>

<body>

3<form action="3.php" method="POST">

Senter username:

<input type="text" name="t1" placeholder="username"><br>

enter password:

<input type="text" name="t2" placeholder="password"><br>

<input type="submit" value="login">

</body>

/htmls




<?php

session_start();

if(!isset($_SESSION['cnt']))

$_SESSION['cnt']=0;

$username=$_POST['t1'];

$password=$_POST['t2'];

if($username=="" && password=="")

{

}

echo"please enter ur username and password";

else if($username=='ty' && password=="123456")

{

echo"login successfull";

$_SESSION['cnt']=0;
}
else

{ $_SESSION['cnt']=$_SESSION['cnt']+1;

if($_SESSION['cnt']>2)

{

echo("you exceed the limit");

$_SESSION['cnt']=0;

}

else

{

echo login failed...wrong details enterd.....attempts made".$_SESSION['cnt'];

Include("3.html")
}
}
?>



<html>

<body>

<form action="4.php" method="PSOT">

<center> ENTER THE DETAILS </center>

enter the employee name:<input type="text" name="t1" placeholder="name"><br>

enter the employee no: <input type="text" name="t2" placeholder="number"> <br>

enter the employee address: <input type="text" name="t3" placeholder="address">

<br>

<input type="submit" value="click">

</form>

</body>

</html>







<?php

session_start();

$en=$_POST[t1];

$enum=$_POST[t2];

$ead=$_POST[t3];

$_SESSION[en]=$en;

$_SESSION[enum]=$enum;

$_SESSION[ead]=$ead;

?>

<html>

<body>

<form action="44.php" method="POST">;

<<center> enter earning details </center>

basic salary:<input type="text" name="e1"><br>
DA:<input type="text" name="e2"><br>

HRA:<input type="text" name="e3"><br>

<input type="submit" value="click"><br>

</form>

</body>
</html>
