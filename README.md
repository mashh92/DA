
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
