
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''

How It Works:
User Inputs:

Principal: The initial deposit amount.
Annual Interest Rate: The annual percentage rate (APR).
Duration: The number of years the CD will be held.
Calculations:

The maturity value is calculated using the formula:
Maturity Value
= Principal × (1 + Annual Rate 100) Years
Maturity Value=Principal×(1+ 100 Annual Rate
 ) Years

User Options:

After showing the maturity value, the chatbot provides options to reinvest or withdraw the amount.
Error Handling:

If the user enters invalid inputs (e.g., non-numeric), it alerts them to correct the input.
'''

def cd_analysis_chatbot():
    print("🤖 Welcome to the CD Maturity Analysis Chatbot!")
    print("I can help you calculate the maturity value of your Certificate of Deposit (CD).")

    # Get user inputs
    try:
        principal = float(input("💰 Enter the initial deposit amount ($): "))
        annual_rate = float(input("📈 Enter the annual interest rate (%): "))
        years = int(input("⏳ Enter the duration of the CD (in years): "))

        # Calculate maturity value
        maturity_value = principal * (1 + (annual_rate / 100)) ** years
        print(f"\n📊 The maturity value of your CD after {years} years will be: ${maturity_value:,.2f}")

        # Provide options upon maturity
        print("\n🔄 What would you like to do upon maturity?")
        print("1. Reinvest the matured amount.")
        print("2. Withdraw the matured amount.")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            print("\n🌱 Reinvesting your matured amount could lead to further growth!")
        elif choice == "2":
            print("\n🏦 You can withdraw the matured amount for personal use or other investments.")
        else:
            print("\n❌ Invalid choice. Please decide wisely upon maturity.")

    except ValueError:
        print("\n⚠️ Invalid input. Please enter numeric values for the deposit, interest rate, and duration.")

    print("\nThank you for using the CD Maturity Analysis Chatbot! 🤖")

# Run the chatbot
cd_analysis_chatbot()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            {
  "chatbot_name": "CD Maturity Analysis Chatbot",
  "description": "A chatbot to calculate the maturity value of a Certificate of Deposit (CD) and provide options upon maturity.",
  "interaction_flow": [
    {
      "step": 1,
      "user_prompt": "Enter the initial deposit amount ($):",
      "input_type": "number",
      "example_input": 10000
    },
    {
      "step": 2,
      "user_prompt": "Enter the annual interest rate (%):",
      "input_type": "number",
      "example_input": 3.5
    },
    {
      "step": 3,
      "user_prompt": "Enter the duration of the CD (in years):",
      "input_type": "number",
      "example_input": 5
    },
    {
      "step": 4,
      "calculation": {
        "formula": "principal * (1 + (annual_rate / 100)) ** years",
        "example_calculation": "10000 * (1 + (3.5 / 100)) ** 5",
        "result": 11876.96
      },
      "response": "The maturity value of your CD after 5 years will be: $11,876.96"
    },
    {
      "step": 5,
      "user_prompt": "What would you like to do upon maturity?",
      "options": [
        {
          "option_id": 1,
          "description": "Reinvest the matured amount.",
          "response": "Reinvesting your matured amount could lead to further growth!"
        },
        {
          "option_id": 2,
          "description": "Withdraw the matured amount.",
          "response": "You can withdraw the matured amount for personal use or other investments."
        }
      ]
    }
  ],
  "error_handling": {
    "invalid_input": "Invalid input. Please enter numeric values for the deposit, interest rate, and duration.",
    "unrecognized_option": "Invalid choice. Please decide wisely upon maturity."
  },
  "completion_message": "Thank you for using the CD Maturity Analysis Chatbot!"
}

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    