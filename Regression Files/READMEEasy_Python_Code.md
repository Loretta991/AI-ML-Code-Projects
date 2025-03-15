
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#1. Write a program that open an output file with the 
#   filename my_name.txt, writes your name to the file then closes the file
my_file = open("my_name.txt", "wt")
my_file.write('Loretta')
my_file.close()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#2. Write a program that open a file created in 1, reads name from my_name.txt
#   displays name on screen then closes the file
my_file = open("my_name.txt", "rt")
my_name = my_file.read()
print('Reading: ', my_name)
my_file.close()  
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #3. Write code that : opens an output file with name number_list.txt,
#   with a loop to write numbers 1 through 100, then close the file
import numpy as np
numFile = open("number_list.txt","rt")
numList = numFile.read()
n = 1 
for i in range(0,101):
    index = 0
    i = i - n
    while n < 1:
     numFile.close()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            

#4. Write code to open number_list.txt and displays all
#   the numbers numbers 1 through 100, then close the file
#   numFile = open("number_list.txt","w")
import numpy as np
numFile = open("number_list.txt","w")
numList = numFile
n = 1 
for i in range(0,101):
    index = 0
    print(i)
    i = i -n
    while n < 1:
     numList.write('{}n'.format(i))
     numFile.close()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#5. count how many words in a string
#   import the string and split the words using split()
#   save in dictionary
fileName = open("text.txt", "rt")
test_file = fileName.read()
words = test_file.split()
for i in range(1):
  index = 0
print(test_file)
i = i - n
while n < 1:
  words = len(test_file.split())
print("The number of words in the string are :",  + len(words))

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #6. Pandas Series - Write python code to create series from a list.
import pandas as pd
# default index ranges is from 0 to len(list) - 1
serList = pd.Series(['apple', 'orange', 'cherry']) 
print(serList)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #7. Combine two Pandas Series ser1 and ser2 from a data frame.
# import pandas library
import pandas as pd
ser1 = pd.Series(["red", "green",
               "yellow"])
ser2 = pd.Series(["blue", "orange",
               "purple"])
# combine two series then show data frame
df = pd.DataFrame(ser1.append(ser2,
     ignore_index = True))
# show the dataframe
df
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #8. Write code to get the information of the data frame in 6.
import pandas as pd
# default index ranges is from 0 to len(list) - 1
serList = pd.Series(['apple', 'orange', 'cherry']) 
print(serList)
df = pd.DataFrame(serList)
# show the dataframe
df
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #9.  Import dirty_data.csv into a data frame
#    replace all missing values with max value of that column.
#    Fix ====> later
import csv
import pandas as pd 
import numpy as np

df = pd.read_csv("dirtydata.csv")
fileName = open("dirtydata.csv", 'r')
n= 1
for i in range(1):
  print('This is the original DirtyData file : ')   
  print(df) 
i = i - n
while n < 1:  
  df.fillna('', inplace = True)
print(df.max(), 'found highesst values')
maxVal = df.max(1)
maxVal2 = df.isnull().max(0)
print(maxVal2 ,'True: Date & Calories have NaN.')
n= 1
for i in range(1):
  index = 0
  i = i - n
while n < 1:
  print ('Number of Na values present: ')
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #10. import the user_data.csv and analyse
#    a) check if any missing data, if yes drop the row
#    b) check correlation of the data
#    c) visualize the data and observ the distribution of the data

import csv
import pandas as pd
import numpy as np
df = pd.read_csv("User_Data.csv")
FileName = open("User_Data.csv",'r')
new_df = df.dropna()
#print(new_df.to_string())
n = 1
for i in range(1):
  index = 0
print('This is the original User_Data file: ')
print(df)
i = i - n
while n < i:

  print('No Data')

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #11.  for the code below
def my_function(food,number):

  food = ("orange","banana","cherry")
  number = 10
  print(food(number))
#  position 10 "banana, a" gets printed  
 
  number = 1
  fruits = ("apple","banana","cherry")

  my_function(fruits,10)
  print(fruits)
  print(number)

  print('What is the out put and why?')
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # 11. Replace the space in my_str with the least frequent character
#     my_str='abc deb abd ggade', 'c' is the least frequent character, so
#     replace the space with 'c'
 
import pandas as pd
my_str = 'abc deb abd ggade'
print("Input series:")
print(my_str)
my_str = pd.Series(list(my_str))
freqElem = my_str.value_counts()
print(freqElem)
curElem = freqElem.dropna().index[-1]
result = "".join(my_str.replace(' ', curElem))
print(result)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from IPython.utils.capture import capture_output
from pandas.core.arrays.interval import value_counts
#12. ser1 = pd.Series([1,2,3,4,5])
#    ser2 = pd.Series([4,5,6,7,8])
#  Get all common items in ser1 and ser2.
import pandas as pd
import numpy as np
ser1 = ([1,2,3,4,5])
ser2 = ([4,5,6,7,8])
print ('Input series:', ser1 , ser2) 
my_ser1 = pd.Series(list(ser1))
my_ser2 = pd.Series(list(ser2))
freqElem =(my_ser1.value_counts() + my_ser2.value_counts())
print(freqElem)

newFreq = freqElem.dropna().index[-1]
newFreqElem = freqElem.value_counts()
result = "".join(newFreqElem.replace('', newFreq))
n = 1
for i in range(1):
  index = 0
print(result)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #13. Randum Number Guessing Game
#Write a program that generates a random num (1,100).
# create prompt to guess what the number is
#display "Too High, try again' or "To Low, try again"
#congatulate for correct guesses and start game over
#Optiona Enhancement: keep count of number of guesses,
# display when the guess correct
import csv
import numpy as np
import pandas as pd
import random
#local test to see if works
rNum = 46
#rNnum = random.NumRange(1,100)
yourGuess = int(input("Guess a number betweem 1 and 100: "))
while yourGuess != rNum:
   if yourGuess < rNum:
     print("Too Low, try again")
     yourGuess = int(input("\nGuess a number between 1 and 100: "))
   else:
      print("Too High, try again")
      yourGuess = int(input("\nGuess a number between 1 and 100: "))
print("Congratulations, Your Guess is correct!  ")
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #14. Create a data frame, and reverse all the rows of a data frame df.
# Here we are just resetting the indexing for the entire database
# and reversing the database.
# Import pandas library
import pandas as pd
  
# initialize list of lists
data = [[112295, 9358, 21370, 90925], [78655, 6555, 23596, 55058], [87544, 7295, 26263, 61280]]
  
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Yearly', 'Monthly', '(Tax)', 'Net'])
# print dataframe.
print('Input: Assumes 30% Tax Bracket')
print (' ')
print(df)
df = pd.DataFrame(data)
print(' ')
# show the dataframe
df

#reversing the database.
df = pd.DataFrame(data, columns=['Yearly', 'Monthly', 'Tax', 'Net'])
df = df.loc[::-1].reset_index(drop=True).head()
print('Reversed:')
print (' ')
print(df)

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    