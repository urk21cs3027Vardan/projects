function generatePassword(){
    let password="";
    let allowed="";
    let UpperCase="ABCDEFGHIJKLMNOPQRSTUVVWXYZ";
    let LowerCase="abcdefghijklmnopqrstuvwxyz";
    let Numbers="1234567890";
    let Symbols="!@#$%^&*";

    let pwdlength=document.getElementById("pwdlength").value;
    let uppercase=document.getElementById("uppercase");
    let lowercase=document.getElementById("lowercase");
    let numbers=document.getElementById("numbers");
    let symbols=document.getElementById("symbols");

        if(uppercase.checked===true){
            allowed=allowed+UpperCase;
        }
        if(lowercase.checked===true){
            allowed=allowed+LowerCase;
        }
        if(numbers.checked===true){
            allowed=allowed+Numbers;
        }
        if(symbols.checked===true){
            allowed=allowed+Symbols;
        }
        for(let i=0;i<pwdlength;i++){
            index=Math.floor(Math.random()*allowed.length);
            password=password+allowed[index];
        }
        document.getElementById("pwd").textContent=password;
    }


