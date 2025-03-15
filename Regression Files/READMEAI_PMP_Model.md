
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Define the stages of the project
stages = {
    "Request for Information": ["Receive Request", "Gather Requirements", "Define Objectives", "Deliver Project Proposal"],
    "Project Sizing": ["Scope and Define Project", "Identify Resources", "Assign Roles and Responsibilities", "Approve Project Plan"],
    "Hardware and Software Considerations": ["Identify Technical Requirements", "Evaluate Solutions", "Select Vendors and Hardware", "Procure Hardware and Software"],
    "Design Phase": ["Create Design Plan", "Develop Architecture", "Create Functional Specifications", "Review and Approve Design"],
    "Coding Phase": ["Develop Software Components", "Test Software Components", "Debug and Refine", "Complete Software Build"],
    "Testing Phase": ["Create Test Plan", "Execute Test Cases", "Identify and Log Defects", "Approve Software for Deployment"],
    "Implementation and Signoff": ["Deploy Software Solution", "Obtain Stakeholder Approval", "Complete User Training", "Signoff and Closeout Project"]
}

# Define a function to display the stages and allow user input
def display_stages():
    print("Project Stages:")
    for i, stage in enumerate(stages):  
         print(f"{i+1}. {stage}")           
    choice = int(input("Enter the stage number to display details: "))           
    if choice < 1 or choice > len(stages):
            print("Invalid choice. Please try again.")       
    else:         
      display_stage_details(stages[list(stages.keys())[choice-1]])

# Define a function to display the details of a stage and allow user input
def display_stage_details(stage):
    for i, task in enumerate(stage):
        print(f"{i+1}. {task}")
    choice = int(input("Enter the task number to add details: "))
    if choice < 1 or choice > len(stage):
        print("Invalid choice. Please try again.") 
        display_stage_details(stage)       
    else:
        add_task_details(stage[choice-1])

# Define a function to add details to a task
def add_task_details(task):
    print(f"\n{task.upper()}")
    details = input("Enter task details: ")
    # save task details to project management template

# Main program loop
while True:
    display_stages()
    choice = input("Do you want to continue (y/n)? ")
    if choice.lower() == "n":
        break
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    