sumuj = function(a, b){
  s = a+b
  return(s)
}

losuj = function(a, b){
  l <- sample(a:b, 1)
  return(l)
}

standaryzuj = function(v){
  return((v-mean(v))/sd(v))
}