import streamlit as st
import pandas as pd
import openai
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Data Analysis Assistant", layout="wide")

# Set your OpenAI API key directly here
try:
    openai.api_key = st.secrets["openai"]["api_key"]
# Fall back to environment variable
except:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

# Function to check if API key is valid
def is_api_key_valid():
    try:
        # Make a minimal API call to check if key is valid
        openai.models.list()
        return True
    except Exception as e:
        st.error(f"API Key Error: {str(e)}")
        return False

# Add title and description
st.title("Data Analysis Assistant")
st.markdown("Upload a CSV file and ask questions about your data to get Python code and results.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose an OpenAI model",
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0
)

# Initialize session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Display dataframe preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Display dataframe info
        st.subheader("Data Information")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
    except Exception as e:
        st.error(f"Error: {e}")

# Query input
query = st.text_area("What would you like to know about the data?", height=100)
submit_button = st.button("Generate Analysis")

# Function to get code from OpenAI
def get_code_from_openai(query, df, model):
    try:
        # Create a prompt that includes information about the DataFrame
        system_prompt = f"""You are an expert data scientist with strong Python and pandas skills. 
        You are given access to a DataFrame with the following columns: {list(df.columns)}. 
        The user may ask any question related to this data.
        You need to just give the Python code and no other thing is required. Don't include ```python and ``` in start and end.

       IMPORTANT LIBRARIES AVAILABLE:
     - pandas (as pd)
     - matplotlib.pyplot (as plt)
     - seaborn (as sns)
     - numpy (as np)
     - plotly.express (as px)
     - plotly.graph_objects (as go)

     Always import these libraries at the beginning of your code if you need them.

     IMPORTANT: If the user requests both table and visualization:
     1. First compute the result DataFrame and assign it to a variable
     2. Then create the visualization
     3. Include BOTH the DataFrame variable name AND the plt.gcf() or fig at the end of your code

     For example:
     ```
     import pandas as pd
     import matplotlib.pyplot as plt

      # Compute the result
     result_df = df.groupby('Product')['Revenue'].mean().reset_index()

     # Create visualization
     plt.figure(figsize=(10, 6))
     plt.bar(result_df['Product'], result_df['Revenue'])
     plt.title('Average Revenue by Product')
     plt.xlabel('Product')
     plt.ylabel('Average Revenue')
     plt.xticks(rotation=45)
     plt.tight_layout()

     # Return both results
     
     plt.gcf()
     result_df
     ```

     This ensures both the table and visualization will be displayed properly."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function to execute the generated code with improved result handling
def execute_code(code, df):
    try:
        # Create a local namespace with necessary libraries and dataframe
        local_namespace = {
            "df": df, 
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "np": np,
            "px": px,
            "go": go
        }
        
        # Execute the code
        exec(code, globals(), local_namespace)
        
        # Try to get the result from the last line
        #lines = code.strip().split("\n")
        #last_line = lines[-2:].strip()
        lines = code.strip().split("\n")
        last_line = [line.strip() for line in lines[-2:]]
        last_line = "\n".join(line.strip() for line in lines[-2:])


        
        # Check if the last line is just a variable name
        if last_line in local_namespace and not last_line.startswith(("import", "from", "#", "df[", "print")):
            return local_namespace[last_line]
        else:
            # Try to evaluate the last line
            try:
                result = eval(last_line, globals(), local_namespace)
                return result
            except:
                # If we can't get the result from the last line, check for specific visualization objects
                
                # Check for matplotlib figure
                if 'plt' in local_namespace and plt.get_fignums():
                    return plt.gcf()
                
                # Check for user-defined variables
                for var_name in local_namespace:
                    # Skip built-in variables and modules
                    if (var_name not in globals() and 
                        not var_name.startswith('__') and 
                        not isinstance(local_namespace[var_name], type(pd))):
                        # Return the first user-defined variable we find
                        if var_name not in ['df', 'plt', 'sns', 'np', 'px', 'go']:
                            return f"{var_name}: {local_namespace[var_name]}"
                
                return "Code executed successfully, but no explicit result to display."
    
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Process the query when the submit button is clicked
if submit_button and query and st.session_state.df is not None:
    # Check if API key is valid before proceeding
    if is_api_key_valid():
        with st.spinner("Generating analysis..."):
            code = get_code_from_openai(query, st.session_state.df, model_option)
            
            # Display the generated code
            st.subheader("Generated Python Code")
            st.code(code, language="python")
            
            # Execute the code and display the result
            with st.spinner("Executing code..."):
                result = execute_code(code, st.session_state.df)
                
                st.subheader("Result")
                
                # Handle different types of results
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, pd.Series):
                    st.dataframe(result.to_frame())
                elif str(type(result)) == "<class 'matplotlib.figure.Figure'>":
                    st.pyplot(result)
                elif 'plotly.graph_objects' in str(type(result)):
                    st.plotly_chart(result)
                elif str(type(result)).startswith("<class 'plotly"):
                    st.plotly_chart(result)
                else:
                    st.write(result)
    else:
        st.error("Please check your OpenAI API key and try again.")
elif submit_button and not st.session_state.df:
    st.error("Please upload a CSV file first.")
elif submit_button and not query:
    st.error("Please enter a query about the data.")

# Instructions in the sidebar
st.sidebar.markdown("## How to Use")
st.sidebar.markdown("""
1. Upload a CSV file
2. Type your question about the data
3. Click 'Generate Analysis'

Example questions:
- Show me the average revenue by product
- Create a bar chart of sales by category
- What's the correlation between price and quantity?
- Show me a pie chart of customer distribution by region
""")

# Footer
st.markdown("---")
st.markdown("Data Analysis Assistant powered by OpenAI and Streamlit")
