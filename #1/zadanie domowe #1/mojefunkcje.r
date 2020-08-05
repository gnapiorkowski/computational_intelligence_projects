sumuj <- function(a,b){
  s <- a+b
  return(s)
}

losuj = function(a,b){
  vectorx = a:b
  return(sample(vectorx, 1))
}
standaryzuj = function(v){
  return((v-mean(v))/sd(v))
}
normalizuj = function(v){
  return((v-min(v))/(max(v)-min(v)))
}
wyszukaj = function(v, x){
  s <- 0
  for (item in v){
    if (item > x){
      s = s + 1
    }
  }
  return(s)
}
wyszukaj2 = function(v, x){
  return(length(v[v>x]))
}
