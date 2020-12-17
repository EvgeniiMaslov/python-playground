best <- function(state, outcome){
  df <- read.csv("hospital_rates/outcome-of-care-measures.csv", colClasses = "character")
  
  lst <- list("heart attack"=11, "heart failure"=17, "pneumonia"=23)
  
  state <- df[df["State"] == state, ]
  
  if(!is.null(lst[outcome]) & nrow(state) > 0){
    
    outc <- lst[[outcome]]
    
    subset <- state[, c(2, outc)]
    
    subset[, 2] <- as.numeric(subset[, 2])
    subset <- subset[complete.cases(subset),]
    names(subset)[2] <- "mortality"
    sorted <- subset[order(subset$mortality, subset$Hospital.Name), ]
    
    return(sorted[1, "Hospital.Name"])
  }
  return("Nothing")
}