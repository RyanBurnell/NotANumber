##############################################################################
#
#     Not a Number: Identifying Instance Features for Capability-Oriented Evaluation
#     IJCAI 2022
#
##############################################################################
# 
# This is R code for analyses from the paper
#
# This code has been developed by
# Ryan Burnell, Jose Hernandez-Orallo, and John Burden
#
# LICENCE:
#   GPL
#
#
##############################################################################

#Read in necessary packages
if (!require('psych')) {  
  install.packages('psych')
  require(psych)
}  

if (!require('mirt')) { 
  install.packages('mirt')
  require(mirt)
}

if (!require('readxl')) { 
  install.packages('readxl')
  require(readxl)
}

if (!require('tidyverse')) { 
  install.packages('tidyverse')
  require(tidyverse)
}

if (!require('lme4')) { 
  install.packages('lme4')
  require(lme4)
}

if (!require('C50')) { 
  install.packages('C50')
  require(C50)
}  

if (!require('randomForest')) { 
  install.packages('randomForest')
  require(randomForest)
} 

if (!require('ggcorrplot')) { 
  install.packages('ggcorrplot')
  require(ggcorrplot)
}  

#Set variables for PDF outputs
PDFEPS = 1 # 0 None, 1 PDF, 2 EPS
PDFheight= 8 # 7 by default, so 14 makes it double higher than wide, 5 makes letters bigger (in proportion) for just one plot
PDFwidth= 8 # 7 by default

#Function to start writing pdf with given dimensions
OpenPDFEPS = function(file, PDFheight=PDFheight, PDFwidth=PDFwidth) {
  if (PDFEPS == 1) {
    #pdf(paste(OUTPUTDIR, "/", file, ".pdf", sep=""), height= PDFheight, width= PDFwidth)
    pdf(paste(file, ".pdf", sep=""), height= PDFheight, width= PDFwidth)
  } else if (PDFEPS == 2) {
    postscript(paste(file, ".eps", sep=""), height= PDFheight, width= PDFwidth, horizontal=FALSE)
  }
}

#Function to end writing of PDF
ClosePDFEPS = function() {
  if (PDFEPS != 0)
    dev.off()
}





#######################
#### Data cleaning ####
#######################

# Read in csv file containing list of tasks getting at 
# goal-oriented action & coded features for each task
rawTaskAnnotatedData = read.csv("task_codes.csv", check.names = FALSE)

#Get rid of coded but unusable tasks (right now only the moving trials)
rawTaskAnnotatedData = filter(rawTaskAnnotatedData, usable == 1)

#clean up column data types
rawTaskAnnotatedData$reward_size = as.numeric(rawTaskAnnotatedData$reward_size)
rawTaskAnnotatedData$reward_distance = as.numeric(rawTaskAnnotatedData$reward_distance)

#copy data for modification to preserve unscaled data
taskAnnotatedData = rawTaskAnnotatedData

#scale variables

#normalise then log transform reward size
taskAnnotatedData$reward_size = -scale(taskAnnotatedData$reward_size) #initial scale, reversing direction so larger values = smaller size (more difficult)
x= taskAnnotatedData$reward_size
x = -log(-x+1) #log transform
taskAnnotatedData$reward_size = x
taskAnnotatedData$reward_size = scale(taskAnnotatedData$reward_size)  # Need to rescale again

#normalise reward distance
taskAnnotatedData$reward_distance = scale(taskAnnotatedData$reward_distance)

#set factor levels manually to get rid of "random" value and ensure correct ordering for regression
taskAnnotatedData$reward_colour = factor(taskAnnotatedData$reward_colour, levels = c("green", "yellow"))
taskAnnotatedData$reward_side = factor(taskAnnotatedData$reward_side,levels = c("straight", "left", "right", "behind"))
taskAnnotatedData$reward_side = addNA(taskAnnotatedData$reward_side)
taskAnnotatedData$facing_reward = factor(taskAnnotatedData$facing_reward, levels = c(0,1))
taskAnnotatedData$facing_reward = addNA(taskAnnotatedData$facing_reward)
taskAnnotatedData$lights = factor(taskAnnotatedData$lights, levels = c("on", "alternating", "off-initially"))

#Create xpos variable representing which side the reward is on, 0 = centre, -1 = left, 1 = right
taskAnnotatedData$reward_Xpos = rep(0, nrow(taskAnnotatedData))
taskAnnotatedData$reward_Xpos[taskAnnotatedData$reward_side == "left"] = -1
taskAnnotatedData$reward_Xpos[taskAnnotatedData$reward_side == "right"] = 1

#Create ypos variable representing whether the reward is in front or behind, NA = random or unclear, -1 = behind, 1 = in front
taskAnnotatedData$reward_Ypos = rep(NA, nrow(taskAnnotatedData))
taskAnnotatedData$reward_Ypos[taskAnnotatedData$reward_side == "behind"] = 1
taskAnnotatedData$reward_Ypos[taskAnnotatedData$reward_side == "straight" | taskAnnotatedData$reward_side == "left" | taskAnnotatedData$reward_side == "right"] = -1

#Read in AnimalAI reward results. Check names = FALSE to avoid R prepending X to column names that start with a number
rewardData = read.csv("rewards_all.csv", check.names = FALSE)
#Narrow down to tasks we annotated to include
rewardData = select(rewardData, team, taskAnnotatedData$task)
#Pivot to get long form dataset
rewardDataLong = pivot_longer(rewardData, cols = -team, names_to = "task", values_to = "reward")
#Merge performance data with task data
rewardDataLong = inner_join(y = taskAnnotatedData, x = rewardDataLong)

# Read in AnimalAI pass/fail results
passFailData = read.csv("scores_all.csv", check.names = FALSE)
#narrow down to tasks from annotated data
passFailData = select(passFailData, team, taskAnnotatedData$task)
#Pivot to get long form dataset
passFailDataLong = pivot_longer(passFailData, cols = -team, names_to = "task", values_to = "passedTrial")
#set pass/fail as boolean
passFailDataLong$passedTrial = as.logical(passFailDataLong$passedTrial)
#Merge performance data with task data
passFailDataLong = inner_join(y = taskAnnotatedData, x = passFailDataLong)

#Aggregate scores together to get mean overall performance for each team
teamScores = aggregate(passedTrial ~ team, passFailDataLong, mean)
#Sort the data from highest score to lowest
teamScores = teamScores[sort(teamScores$passedTrial, decreasing=TRUE, index.return=TRUE)$ix,]

##################
#### Analysis ####
##################

# Global variable to change whether plots include a selection of all agents, the best, the worst or just one
SELECTION = "ALL"  # "ALL" or "BEST" or "WORST" or "ONE"
#Percentile for "BEST/WORST" selection to select based on
percentile = 0.5 # 50%
#chosen agent if "ONE" selected
chosen = "ironbar"
#Toggle whether characteristic grids should group instances into bins
groupedCharacteristicGrids = TRUE

#Narrow passFailDataLong dataset based on chosen subset of agents
if (SELECTION == "BEST") {
  sel = passFailDataLong$team %in% teamScores[1:ceiling(nrow(teamScores)*percentile),]$team
  passFailDataLong = passFailDataLong[sel, ]
  teamsSelected = paste0("Best ", percentile*100, "% of teams")
} else if (SELECTION == "WORST") {
  sel = passFailDataLong$team %in% teamScores[ceiling(nrow(teamScores)*(1-percentile)):nrow(teamScores),]$team
  passFailDataLong = passFailDataLong[sel, ]
  teamsSelected = paste0("Worst ", percentile*100, "% percentile of teams")
} else if (SELECTION == "ONE") {
  sel = passFailDataLong$team %in% chosen
  passFailDataLong = passFailDataLong[sel, ]
  teamsSelected = paste0("Team ", chosen)
} else {
  teamsSelected = "All teams"
}


####Plots####

#Start writing PDF
OpenPDFEPS(paste0("Correlations"), PDFheight=4, PDFwidth=6)

# Plot correlations between variables
##Create subset of columns to correlate
correlationData = select(passFailDataLong, passedTrial, reward_size, reward_distance, reward_colour, reward_Xpos, reward_Ypos)
##Make sure reward colour is numeric for calculation of correlations
correlationData$reward_colour = as.numeric(correlationData$reward_colour)
##Plot correlations
ggcorrplot(corr = cor(correlationData, use= "pairwise.complete.obs", method = "spearman"),method = "square",show.diag = F, type="upper", lab=TRUE, lab_size=3, colors = c("blue", "white", "red3"), ggtheme = theme_classic(), outline.color = "black",tl.cex = 12) +
  scale_x_discrete(labels = c("Passed\ntrial", "Size", "Distance", "Color", "XPos"))+
  scale_y_discrete(labels = c("Size", "Distance", "Color", "XPos", "YPos"))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), text = element_text(size = 10))#, panel.background = element_rect(fill = 'lightgrey '))

#End writing of PDF  
ClosePDFEPS()


#Start writing PDF
OpenPDFEPS(paste0("Curve_Reward_Size"), PDFheight=5, PDFwidth=6)

#Plot of success rate by reward size
##Set variables to plot
x= passFailDataLong$reward_size # reward size
y= (TRUE == passFailDataLong$passedTrial) #true false values of trial pass/fail

##plot variables
plot(x,y, xlab="Reward Size (Large to Small)", ylab="Success Rate", cex.lab=1.5) #main=paste("Success rate by reward size for", teamsSelected),)
##add linear regression line
mod = lm(y ~ x)
abline(mod, col="black", lty=2)

##add points and line taking mean success rate at each value of reward size
xu = sort(as.vector(unique(x)))
yu = NULL
for (i in 1:length(xu)) {
 yu[i] = mean(TRUE == y[x == xu[i]], na.rm=TRUE)
}  
lines(xu, yu, col="blue", type="b", lwd=2, pch=15)
mod = lm(yu ~ xu)
abline(mod, col="blue", lty=2)

#End writing PDF
ClosePDFEPS()

#Start writing PDF

OpenPDFEPS(paste0("Curve_Reward_Distance"), PDFheight=5, PDFwidth=6)
#Plot of success rate by reward distance
##Set variables to plot
x= passFailDataLong$reward_distance
y= (TRUE == passFailDataLong$passedTrial)
##plot variables
plot(x,y, xlab="Reward Distance (Short to Long)", ylab="Success Rate", cex.lab=1.5)#main=paste("Success rate by reward distance for", teamsSelected))
##add linear regression line
mod = lm(y ~ x)
abline(mod, col="black", lty=2)

##add points and line taking mean success rate at each value of reward size
xu = sort(as.vector(unique(x)))
yu = NULL
for (i in 1:length(xu)) {
  yu[i] = mean(TRUE == y[x == xu[i]], na.rm=TRUE)
}  
lines(xu, yu, col="blue", type="b", lwd=2, pch=15)
mod = lm(yu ~ xu)
abline(mod, col="blue", lty=2)

#End writing PDF
ClosePDFEPS()

#Start writing PDF
OpenPDFEPS(paste0("Curve_Ypos"), PDFheight=5, PDFwidth=6)

#plot of ypos success rates
x= passFailDataLong$reward_Ypos
y= (TRUE == passFailDataLong$passedTrial)
plot(x,y, xlab="Ypos (Front to Behind)", ylab="Success Rate", cex.lab=1.5)#%main=teamsSelected)
mod = lm(y ~ x)
abline(mod, col="black", lty=2)
xu = sort(as.vector(unique(x)))
yu = NULL
for (i in 1:length(xu)) {
  yu[i] = mean(TRUE == y[x == xu[i]], na.rm=TRUE)
}  
lines(xu, yu, col="blue", type="b", lwd=2, pch=15)
mod = lm(yu ~ xu)
abline(mod, col="blue", lty=2)

#End writing PDF
ClosePDFEPS()

#Start writing PDF
OpenPDFEPS(paste0("Curve_Xpos"), PDFheight=5, PDFwidth=6)

#plot of xpos success rates
x= passFailDataLong$reward_Xpos
y= (TRUE == passFailDataLong$passedTrial)
plot(x,y, xlab="Xpos (Left to Right)", ylab="Success Rate", cex.lab=1.5)#, main=teamsSelected)
mod = lm(y ~ x)
abline(mod, col="black", lty=2)
xu = sort(as.vector(unique(x)))
yu = NULL
for (i in 1:length(xu)) {
  yu[i] = mean(TRUE == y[x == xu[i]], na.rm=TRUE)
}  
lines(xu, yu, col="blue", type="b", lwd=2, pch=15)
mod = lm(yu ~ xu)
abline(mod, col="blue", lty=2)

#End writing PDF
ClosePDFEPS()



####Agent Characteristic Grids####

#MAKE SURE SELECTION IS SET TO "ALL" WHEN RUNNING CHARACTERISTIC GRID FUNCTIONS

#Function to draw CharacteristicGrid of data based on two dimensions.
#Takes as arguments two predictor variables (x1, x2) as well as pass fail data for the trials (y)
#Also takes the name of the agent(s) and optional arguments specifying how the data should be grouped

#TODO add parameters for axis label names
makeCharacteristicGrid = function(x1, x2, y, agentName, grouped = TRUE, x1Breaks = NULL, x2Breaks = NULL, x1Name= NULL, x2Name= NULL){
  
  #if we want to group close values together, set up groupings
  if(grouped){
    if(!is.null(x1Breaks)){
      x1 = cut(x1, breaks = x1Breaks)
    }
    if(!is.null(x2Breaks)){
      x2 = cut(x2, breaks = x2Breaks)
    }
  }
  
  #aggregate data together for plotting
  x1 = data.frame(x1)
  x2 = data.frame(x2)
  aggregatedData = data.frame(cbind(x1,x2,y))
  meanData = aggregate(y ~ x1 + x2, aggregatedData, mean)#calculate mean for each group/point
  lengthData = aggregate(y ~ x1+x2, aggregatedData, length)#calculate number of trials at each point
  
  mycol = NULL
  for (i in 1:nrow(meanData)) { 
    mycol[i] = rgb(1-meanData$y[i], meanData$y[i], 0, 0.5) ## red green blue alpha
  }
  fname = paste0("IndividualGrids/", agentName)
  #png(fname)
  if(!grouped){
    #    plot = plot(meanData$x1,meanData$x2, xlab="Reward Size (Large to Small)", ylab="Reward Distance (Short to Long)", col=mycol, pch=15, cex=2, main=name)
    plot = plot(meanData$x1,meanData$x2, xlab=x1Name, ylab=x2Name, col=mycol, pch=15, cex=2, main=name)
  }
  #dev.off()
  #exit()
  meanData$len = lengthData$y
  capability = sum(meanData$y)/nrow(meanData)
  if(grouped){
    plot = ggplot(meanData, aes(x1, x2, fill=y)) +
      geom_tile(aes(alpha=len), color="black", size=0.5) +
      labs(x=x1Name, y=x2Name, fill="Success Rate", alpha="Count") +
      #labs(x=x1Name, y=x2Name, fill="Success Rate", alpha="Count") +
      scale_fill_gradient(low="red", high="green", limits=c(0,1)) +
      scale_alpha(range=c(0.4,0.9)) +
      theme_bw() +
      theme_minimal(base_size=18) +
      geom_text(aes(label = len)) +
      annotate("text",label=paste("Capability: ", capability),x=1.2, y=0.75)+
      coord_cartesian(clip="off")
  }
  return(plot)
  # print(plot)
}

#function to save characteristic grids for all agents, given specified variable names and breaks for the plot
saveAllCharacteristicGridsPassFail = function(teamNames, data, grouped = TRUE, x1Name, x2Name, x1Breaks = NULL, x2Breaks = NULL, folderName){
  #iterate through teams, saving characteristic grids
  
  levels(teamNames)=c(levels(teamNames),"All")
  teamNames[length(teamNames)+1]="All"
  for(t in teamNames){
    #select only rows with the given team name
    if ( t != "All"){
      sel = data$team %in% t
      gridData = data[sel, ]
    } 
    else {
      gridData = data
    } 
    #specify two features to look at
    x1 = as.matrix(select(gridData, x1Name))
    x2 = as.matrix(select(gridData, x2Name))
    
    #specify pass/fail column
    y = (gridData$passedTrial == TRUE)
    
    #run function to make characteristic grids using the specified parameters
    newPlot = makeCharacteristicGrid(x1 = x1, x2 = x2, y = y, agentName = t, grouped = grouped, x1Breaks, x2Breaks, x1Name, x2Name)
    
    #save plot to file
    # the size of the plot depends on the size of the "Plots" screen. It should be better to generate PDFs and set the size fixed.
    fileName = paste0(t, ".png")
    ggsave(fileName, plot = newPlot, path = folderName, width=24, height=16, unit="cm")
  }
}

#function to save characteristic grids for all agents, given specified variable names and breaks for the plot
saveAllCharacteristicGridsPassFail = function(teamNames, data, grouped = TRUE, x1Name, x2Name, x1Breaks = NULL, x2Breaks = NULL, folderName){
  #iterate through teams, saving characteristic grids
  
  levels(teamNames)=c(levels(teamNames),"All")
  teamNames[length(teamNames)+1]="All"
  for(t in teamNames){
    #select only rows with the given team name
    if ( t != "All"){
      sel = data$team %in% t
      gridData = data[sel, ]
    } 
    else {
      gridData = data
    } 
    #specify two features to look at
    x1 = as.matrix(select(gridData, x1Name))
    x2 = as.matrix(select(gridData, x2Name))
    
    #specify pass/fail column
    y = data$reward
    
    #run function to make characteristic grids using the specified parameters
    newPlot = makeCharacteristicGrid(x1 = x1, x2 = x2, y = y, agentName = t, grouped = grouped, x1Breaks, x2Breaks, x1Name, x2Name)
    
    #save plot to file
    # the size of the plot depends on the size of the "Plots" screen. It should be better to generate PDFs and set the size fixed.
    fileName = paste0(t, ".png")
    ggsave(fileName, plot = newPlot, path = folderName, width=24, height=16, unit="cm")
  }
}

#function to save characteristic grids for all agents, given specified variable names and breaks for the plot
saveAllCharacteristicGridsPassFail = function(teamNames, data, grouped = TRUE, x1Name, x2Name, x1Breaks = NULL, x2Breaks = NULL, folderName){
  #iterate through teams, saving characteristic grids
  
  levels(teamNames)=c(levels(teamNames),"All")
  teamNames[length(teamNames)+1]="All"
  for(t in teamNames){
    #select only rows with the given team name
    if ( t != "All"){
      sel = data$team %in% t
      gridData = data[sel, ]
    } 
    else {
      gridData = data
    } 
    #specify two features to look at
    x1 = as.matrix(select(gridData, x1Name))
    x2 = as.matrix(select(gridData, x2Name))
    
    #specify pass/fail column
    y = (gridData$passedTrial == TRUE)
    
    #run function to make characteristic grids using the specified parameters
    newPlot = makeCharacteristicGrid(x1 = x1, x2 = x2, y = y, agentName = t, grouped = grouped, x1Breaks, x2Breaks, x1Name, x2Name)
    
    #save plot to file
    # the size of the plot depends on the size of the "Plots" screen. It should be better to generate PDFs and set the size fixed.
    fileName = paste0(t, ".png")
    ggsave(fileName, plot = newPlot, path = folderName, width=24, height=16, unit="cm")
  }
}

####Capability Metrics####

#Function to compute capability score, given two or three dimensions. 
#Capability score is calculated by taking the average performance of the agent within each "bin" or section of the dimension space, and then taking the mean of each bin. 
#Essentially a weighted average such that each bin is equally weighted
computeCapability = function(x1, x2=NULL, x3 = NULL, y, x1Breaks=NULL, x2Breaks=NULL, x3Breaks=NULL) {
  if(!is.null(x1Breaks)){
    x1 = cut(x1, breaks = x1Breaks)
    x1= data.frame(x1)
  }
  if(!is.null(x2Breaks) & !is.null(x2)){
    x2 = cut(x2, breaks = x2Breaks)
    x2 = data.frame(x2)
  }
  if(!is.null(x3Breaks) & !is.null(x3)){
    x3 = cut(x3, breaks=x3Breaks)
    x3 = data.frame(x3)
  }

  
  if(!is.null(x3)){
    aggregatedData = data.frame(cbind(x1,x2,x3,y))
    meanData = aggregate(y ~ x1 + x2+x3, aggregatedData, mean)
    lengthData= aggregate(y~x1+x2+x3, aggregatedData, length)
  }  else if(!is.null(x2)){
    aggregatedData=data.frame(cbind(x1,x2,y))
    meanData = aggregate(y~x1+x2, aggregatedData, mean)
    lengthData=aggregate(y~x1+x2, aggregatedData, length)
  } else{
    aggregatedData=data.frame(cbind(x1,y))
    meanData = aggregate(y~x1, aggregatedData,mean)
    lengthData =aggregate(y~x1, aggregatedData,length)
  }
  meanData$len = lengthData$y
  capability = sum(meanData$y)/nrow(meanData)
  return(capability)
}

#Function to compute the capability scores for each team in the dataset
computeAllCapability = function(teamNames, data, x1Name, x2Name, x3Name, x1Breaks=NULL, x2Breaks=NULL, x3Breaks=NULL){
  #iterate through the teams in the dataset
  agentCapabilitiesMatrix = data.frame(matrix(list(), nrow=length(teamNames),ncol=8), stringsAsFactors = FALSE)
  colnames(agentCapabilitiesMatrix) = c("x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "agent")
  for(i in 1:length(teamNames)){
    #For each team, get the data for that team
    t = teamNames[[i]]
    sel = data$team %in% t
    gridData = data[sel, ]
    
    #specify the features to look at
    x1 = as.matrix(select(gridData, x1Name))
    x2 = as.matrix(select(gridData, x2Name))
    x3 = as.matrix(select(gridData, x3Name))
    
    #Compute the capability score for each combination of features
    y = (gridData$passedTrial == TRUE)
    agentCapabilitiesMatrix[[i, 1]] = computeCapability(x1=x1, x1Breaks=x1Breaks, y=y)
    agentCapabilitiesMatrix[[i, 2]] = computeCapability(x1=x2, x1Breaks= x2Breaks, y=y)
    agentCapabilitiesMatrix[[i, 3]] = computeCapability(x1=x3, x1Breaks=x3Breaks, y=y)
    agentCapabilitiesMatrix[[i, 4]] = computeCapability(x1=x1,x2=x2, x1Breaks=x1Breaks, x2Breaks=x2Breaks,y=y)
    agentCapabilitiesMatrix[[i, 5]] = computeCapability(x1=x1,x2=x3, x1Breaks=x1Breaks, x2Breaks=x3Breaks,y=y)
    agentCapabilitiesMatrix[[i, 6]] = computeCapability(x1=x2,x2=x3, x1Breaks=x2Breaks, x2Breaks=x3Breaks, y=y)
    agentCapabilitiesMatrix[[i, 7]] = computeCapability(x1=x1,x2=x2,x3=x3, x1Breaks=x1Breaks, x2Breaks=x2Breaks, x3Breaks=x3Breaks, y=y)
    #agentCapabilitiesMatrix$agent[i] =t
  }
  
  #Return a matrix with the various scores for each agent
  agentCapabilitiesMatrix$agent = teamNames
  return(agentCapabilitiesMatrix)
  
  
}

#Compute capability scores for each team based on size, distance, and YPos and print
cam = computeAllCapability(teamNames=teamScores$team, data=passFailDataLong, x1Name="reward_size", x2Name="reward_distance", x3Name="reward_Ypos", x1Breaks=c(-3.5,-2,-0.5, 1, 2.5), x2Breaks = c(-2.5,-1.5,-0.5,0.5,1.5,2.5), x3Breaks= c(-1.5,-0.5,0.5,1.5))
cam

#Print simple average performance and then capability score based on all three features
teamScores[2]
cam[7]

#Plot average performance metric against capability score with line
capPlot = data.frame(x=unlist(teamScores[2]), y=unlist(cam[7]))
capPlot

plot(capPlot$x, capPlot$y, xlab="Success Rate", ylab="Capability")
abline(lm(capPlot$y~capPlot$x), col="red")

#Code to save grids for various combos of features. IMPORTANT: Requires folders with the appropriate names to exist
#save grids for size/distance
saveAllCharacteristicGridsPassFail(teamNames = teamScores$team, data = passFailDataLong, grouped = TRUE,x1Name =  "reward_size", x2Name = "reward_distance", x1Breaks = c(-3.5,-2,-0.5, 1, 2.5), x2Breaks = c(-2.5,-1.5,-0.5,0.5,1.5,2.5), folderName = "Size_Distance_CharacteristicGrids")

#save grids for size/Ypos
saveAllCharacteristicGridsPassFail(teamNames = teamScores$team, data = passFailDataLong, grouped = TRUE,x1Name =  "reward_size", x2Name = "reward_Ypos", x1Breaks =  c(-3.5,-2,-0.5, 1, 2.5), x2Breaks = c(2), folderName = "Size_Ypos_CharacteristicGrids")

#save grids for distance/Ypos
saveAllCharacteristicGridsPassFail(teamNames = teamScores$team, data = passFailDataLong, grouped = TRUE,x1Name =  "reward_distance", x2Name = "reward_Ypos", x1Breaks = c(-2.5,-1.5,-0.5,0.5,1.5,2.5), x2Breaks = c(2), folderName = "Distance_Ypos_CharacteristicGrids")

#save grids of xPos/YPos
saveAllCharacteristicGridsPassFail(teamNames = teamScores$team, data = passFailDataLong, grouped = TRUE,x1Name =  "reward_Xpos", x2Name = "reward_Ypos", x1Breaks =  c(3), x2Breaks = c(2), folderName = "XPos_YPos_CharacteristicGrids")


#Correlations between capability and average performance
##Calculate average performance for each instance
summaryData = aggregate(passFailDataLong$passedTrial, by=list(agent=passFailDataLong$team), FUN=mean)
names(summaryData)[names(summaryData) == 'x'] = 'passRate'
summaryData = merge(summaryData, cam, by = "agent")
##Get capability for each instance
summaryData$capability = as.numeric(as.vector(summaryData$x1x2x3))
##Calculate and plot correlations
cor(summaryData$passRate, summaryData$capability)
plot(summaryData$passRate, summaryData$capability)

#### Predictive models ####

#Decision tree model predicting agent pass/fail
# Train on 75% of the sample size
totalN = nrow(passFailDataLong)
smp_size = floor(0.75 * totalN)

set.seed(1)
train_ind = sample(seq_len(totalN), size = smp_size)

#Set up train and test data and set DV as factor
train = passFailDataLong[train_ind, ]
test = passFailDataLong[-train_ind, ]
train$passedTrial = factor(train$passedTrial)
test$passedTrial = factor(test$passedTrial)

Eval = function(pred, test) {
  acc = mean(pred == test)
  baseline = mean(test == TRUE)
  cat("\nAccuracy:", acc, "with baseline (majority class):", baseline, "\n")
}

Error = function(pred, test) {
  err = mean(pred != test)
  cat("\nCrisp Error:", err, "\n")
} 

MAE = function(pred, test) {
  mae = mean(abs(pred - test))
  cat("\nMAE:", mae, "\n")
}

MSE = function(pred, test) {  # Brier Score
  mse = mean((pred - test)^2)
  cat("\nMSE:", mse, "\n")
}

# #Decision tree model with all features regardless of agent
# passFailC50 = C5.0(formula = passedTrial ~ reward_size + reward_distance + reward_colour + reward_side + facing_reward + lights  + reward_Xpos + reward_Ypos, data = train)
# Eval(predict(passFailC50, test),  test$passedTrial)
# 
# #only relevant features regardless of agent
# passFailC50 = C5.0(formula = passedTrial ~ reward_size + reward_distance + reward_Ypos, data = train)
# Eval(predict(passFailC50, test),  test$passedTrial)

n = nrow(test)

#Error rates for model predicting based on majority class
Error(rep(TRUE,n),  test$passedTrial)  # Majority class error
MAE(rep(1,n),  (test$passedTrial==TRUE)*1.0)  # Majority class MAE
MSE(rep(1,n),  (test$passedTrial==TRUE)*1.0)  # Majority class MSE

unifv = runif(n, 0, 1)

#Error rates for model predicting based on global accuracy across all agents
train_prop = mean(train$passedTrial=="TRUE")
Error(train_prop >= unifv,  test$passedTrial)  # Proportional class error
MAE(rep(train_prop,nrow(test)),  (test$passedTrial==TRUE)*1.0)  # Majority class MAE
MSE(rep(train_prop,nrow(test)),  (test$passedTrial==TRUE)*1.0)  # Majority class MSE

#Error rates for model predicting based on accuracy of each agent
pred = NULL
for (i in 1:nrow(test)) {
  myteam = test[i,]$team
  trainacc = mean(train[train$team == myteam,]$passedTrial == "TRUE")
  # print(trainacc)
  pred[i] = trainacc
}

Error((pred >= unifv), (test$passedTrial=="TRUE")*1)
MAE(pred, (test$passedTrial=="TRUE")*1)
MSE(pred, (test$passedTrial=="TRUE")*1)

#Decision tree model using all features to predict success/failure
passFailC50 = C5.0(formula = passedTrial ~ reward_size + reward_distance + reward_colour + reward_side + facing_reward + lights  + reward_Xpos + reward_Ypos + team, data = train)
Error(predict(passFailC50, test),  test$passedTrial)
MAE(predict(passFailC50, test, type="prob")[,2],  (test$passedTrial==TRUE)*1.0)
MSE(predict(passFailC50, test, type="prob")[,2],  (test$passedTrial==TRUE)*1.0)

#Decision tree model using only relevant features to predict success/failure
passFailC50 = C5.0(formula = passedTrial ~ reward_size + reward_distance + reward_Ypos + team, data = train)
Error(predict(passFailC50, test),  test$passedTrial)
MAE(predict(passFailC50, test, type="prob")[,2],  (test$passedTrial==TRUE)*1.0)
MSE(predict(passFailC50, test, type="prob")[,2],  (test$passedTrial==TRUE)*1.0)


# Function to save a simple plot to show how the predictive model works for a grid of two variables.
# Shades of green/red show the level of confidence.
savePredictionPlot = function(team){
  REP = 75
  test_instance = test[1,]
  test_instance$team = team                    #"y.yang"
  OpenPDFEPS(paste0("Predictions_for_", test_instance$team), PDFheight=5, PDFwidth=6)
  xmin = min(passFailDataLong$reward_size, na.rm=TRUE)
  xmax = max(passFailDataLong$reward_size, na.rm=TRUE)
  ymin = min(passFailDataLong$reward_distance, na.rm=TRUE)
  ymax = max(passFailDataLong$reward_distance, na.rm=TRUE)
  
  #specify bins
  predictedPlotXBreaks = c(-3.5,-2,-0.5, 1, 2.5)
  predictedPlotYBreaks = c(-2.5,-1.5,-0.5,0.5,1.5,2.5)
  
  #draw plot base
  plot(x = c(-3.5,2.5),y =c(-2.5,2.5), col="white", xlab="Reward Size", ylab= "Reward Distance", )+theme_minimal()  #, main=paste0("Predictions for ", test_instance$team))

  #Iterate through bins, drawing the appropriate colour based on the prediction at each bin
  for(i in 1:(length(predictedPlotXBreaks)-1)){
    binXMin = predictedPlotXBreaks[i]#max(predictedPlotXBreaks[i], xmin)
    binXMax = predictedPlotXBreaks[i+1]#min(predictedPlotXBreaks[i+1], xmax)
    #print(paste("X values:", binXMin, ", ", binXMax))
    for(j in 1:(length(predictedPlotYBreaks)-1)){
      #Go through each point within the bin and add to a vector to calculate the average
      points = vector()
      binYMin = predictedPlotYBreaks[j]#max(predictedPlotYBreaks[j], ymin)
      binYMax = predictedPlotYBreaks[j+1]#min(predictedPlotYBreaks[j+1], ymax)
      for(x in seq(binXMin,binXMax,length.out=REP)){
        for (y in seq(binYMin,binYMax,length.out=REP)) {
          #print(paste("x: ", x, ", y:",y ))
          test_instance$reward_size = x
          test_instance$reward_distance = y
          p = predict(passFailC50, test_instance, type="prob")[,2]  # 2 to take the probability for TRUE
          points = append(x = points, values = p)
        }
      }
      #Calculate the average predicted performance in the bin
      meanPrediction = mean(points)
      #Draw the appropriate colour based on the average (more green is better, more red is worse)
      rect(xleft = binXMin, xright = binXMax, ybottom = binYMin,ytop = binYMax, col=rgb(1-meanPrediction, meanPrediction, 0, 0.5))
    }
    
  }
  ClosePDFEPS()
}

OpenPDFEPS()

#save the prediction plot for one agent, in this case y.yang
savePredictionPlot("y.yang")

ClosePDFEPS()


