rankall <- function(outcome, num="best"){
  df <- read.csv("hospital_rates/outcome-of-care-measures.csv", colClasses = "character")
  
  lst <- list("heart attack"=11, "heart failure"=17, "pneumonia"=23)
  
  answ <- data.frame(Hospital.Name=character(),
                     Rate=numeric(),
                     State=character())
  
  states <- unique(df[["State"]]) 
  for(st in states[order(states)]){
  
    state <- df[df["State"] == st, ]
    
    if(!is.null(lst[outcome]) & nrow(state) > 0){
      
      outc <- lst[[outcome]]
      
      subset <- state[, c(2, outc, 7)]
      
      subset[, 2] <- as.numeric(subset[, 2])
      subset <- subset[complete.cases(subset),]
      names(subset)[2] <- "Rate"
      sorted <- subset[order(subset$Rate, subset$Hospital.Name), ]
      
      
      if(num == "best"){
        row <- sorted[1, ]
      }
      
      else if(num == "worst"){
        row <- sorted[nrow(sorted), ]
      }
      else{
        if(num > nrow(sorted)){row <- list("Hospital.Name"=NA, "Rate"=NA, "State"=st)}
        else{row <- sorted[num, ]}
      }
      
    }
    
    answ <- rbind(answ, row)
  }
  return(answ)
}


tail(rankall("pneumonia", "worst"), 10)