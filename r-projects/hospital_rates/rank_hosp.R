rankhospital <- function(state, outcome, num="best"){
  df <- read.csv("hospital_rates/outcome-of-care-measures.csv", colClasses = "character")
  
  lst <- list("heart attack"=11, "heart failure"=17, "pneumonia"=23)
  
  state <- df[df["State"] == state, ]
  
  if(!is.null(lst[outcome]) & nrow(state) > 0){
    
    outc <- lst[[outcome]]
    
    subset <- state[, c(2, outc)]
    
    subset[, 2] <- as.numeric(subset[, 2])
    subset <- subset[complete.cases(subset),]
    names(subset)[2] <- "Rate"
    sorted <- subset[order(subset$Rate, subset$Hospital.Name), ]
    sorted["Rank"] = 1:nrow(sorted)
    
    if(num == "best"){
      return(sorted[1, ])
    }
    
    else if(num == "worst"){
      return(sorted[nrow(sorted), 1])
    }
    else{
      if(num > nrow(sorted)){return(NA)}
      
      return(sorted[num, ])
    }
    
  }
  return("Nothing")
}
