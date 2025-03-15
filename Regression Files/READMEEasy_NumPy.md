
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;"># Numpy</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 1. Import NumPy as np</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 2. Create an array of 10 zeros</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.zeros(10)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 3. Create an array of 10 ones</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.ones(10)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 4. Create an array of 10 fives</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.ones(10) * 5
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 5. Create an array of the integers from 10 to 50</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.arange(10,51)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 6. Create an array of all the even integers from 10 to 50</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.arange(10,51,2)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 7. Create a 3x3 matrix with values ranging from 0 to 8</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.arange(9).reshape(3,3)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 8. Create a 3x3 identity matrix</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.eye(3)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 9. Use NumPy to generate a random number between 0 and 1<br><br>&emsp;NOTE: Your result's value should be different from the one shown below.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.random.rand(1)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 10. Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution<br><br>&emsp;&ensp;NOTE: Your result's values should be different from the ones shown below.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.random.randn(25)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 11. Create the following matrix:</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.arange(1,101).reshape(10,10) / 100
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 12. Create an array of 20 linearly spaced points between 0 and 1:</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">## Numpy Indexing and Selection

Now you will be given a starting matrix (be sure to run the cell below!), and be asked to replicate the resulting matrix outputs:</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # RUN THIS CELL - THIS IS OUR STARTING MATRIX
mat = np.arange(1,26).reshape(5,5)
mat
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 13. Write code that reproduces the output shown below.<br><br>&emsp;&ensp;Be careful not to run the cell immediately above the output, otherwise you won't be able to see the output any more.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat[2:,1:]
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 14. Write code that reproduces the output shown below.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat[3,4]
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 15. Write code that reproduces the output shown below.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat[:3,1:2]
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 16. Write code that reproduces the output shown below.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat[4,:]
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 17. Write code that reproduces the output shown below.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat[3:5,:]
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">## NumPy Operations</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 18. Get the sum of all the values in mat</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat.sum()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 19. Get the standard deviation of the values in mat</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat.std()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### 20. Get the sum of all the columns in mat</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mat.sum(axis=0)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">To insure that we always get the same random numbers</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.random.seed(101)
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    