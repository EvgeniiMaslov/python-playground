pollutantmean <- function(directory, pollutant, id=1:332){
  total_mean <- 0
  total_cnt <- 0
  
  for(i in id){
    index <- as.character(i)
    
    while(nchar(index) < 3){
      index <- paste(0, index, sep="")
    }
    
    filename <- paste(directory, "/", index, ".csv", sep="")
    
    df <- read.csv(filename)
    
    mask <- is.na(df[pollutant])
    total_mean <- total_mean + sum(df[!mask, pollutant])
    total_cnt <- total_cnt + sum(!mask)
    
  }
 
  answ <- total_mean / total_cnt
  
  return(answ)
}


complete <- function(directory, id=1:332){
  
  answ <- data.frame(id=integer(),
                     nobs=integer())
  
  for(i in id){
    index <- as.character(i)
    
    while(nchar(index) < 3){
      index <- paste(0, index, sep="")
    }
    
    filename <- paste(directory, "/", index, ".csv", sep="")
    
    df <- read.csv(filename)
    nobs <- sum(complete.cases(df))
    
    row <- data.frame(i, nobs)
    names(row) <- c("id", "nobs")
    
    answ <- rbind(answ, row)
    
  }
  
  return(answ)
}


corr <- function(directory, threshold=0){
  
  answ <- c()
  
  for(i in list.files(directory)){
    
    filename <- paste(directory, "/", i, sep="")
    
    df <- read.csv(filename)
    
    compl = complete.cases(df)
    
    if(sum(compl) > threshold){
      col <- df[compl,c("sulfate", "nitrate")]
      answ <- c(answ, cor(col["sulfate"], col["nitrate"]))
      
    }
    
    
  }
  return(answ)
}

cr <- corr("specdata", 150)
head(cr)









