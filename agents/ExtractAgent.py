from swarms import Agent
from .config import client
from google.genai import types
import PIL.Image

EXTRACT_PROMPT = """Analyze all charts in the provided image and extract **all possible details**. If multiple charts are present, clearly separate the extracted information for each chart. The output should follow this structure:

## **1. Overview of Charts in the Image**
- **Total Number of Charts**: Identify the number of distinct charts present in the image.
- **Chart Titles (if available)**: List the title of each chart.
- **Chart Types**: Identify the type of each chart (e.g., Bar Chart, Line Chart, Pie Chart, Scatter Plot, etc.).
- **Primary Subject**: Briefly describe what each chart represents.
- **Timeframe (if applicable)**: Mention the period covered by each chart.

## **2. Detailed Analysis for Each Chart**
For each chart, extract the following details:

### **Chart [X]: [Chart Title or Description]**
#### **General Information**
- **Chart Type**: Identify the type of chart.
- **Primary Subject**: What the chart is measuring or analyzing.
- **Legend Information**: List all categories, labels, or series appearing in the chart.

#### **Axis & Data Details**
##### **X-Axis Information**
- **Label**: Extract the title of the X-axis (e.g., 'Months', 'Product Categories').
- **All Values**: List every unique value on the X-axis.

##### **Y-Axis Information**
- **Label**: Extract the title of the Y-axis (e.g., 'Revenue (in USD)', 'Number of Sales').
- **Value Range**:
  - **Minimum Value**: The smallest numerical value.
  - **Maximum Value**: The highest numerical value.
- **All Numerical Values**:
  - **Each Data Point**: List every plotted value along with its corresponding X-axis label.

#### **3. Detailed Numerical Data by Category**
For charts with multiple series/categories, organize data in tables.

##### **Category 1: [Category Name]**
| X-Axis Value  | Y-Axis Value |
|--------------|-------------|
| [Value 1]    | [Number]    |
| [Value 2]    | [Number]    |
| [Value 3]    | [Number]    |
| ...          | ...         |

##### **Category 2: [Category Name]**
| X-Axis Value  | Y-Axis Value |
|--------------|-------------|
| [Value 1]    | [Number]    |
| [Value 2]    | [Number]    |
| [Value 3]    | [Number]    |
| ...          | ...         |

#### **4. Statistical Summary**
- **Minimum, Maximum, and Average Values for Each Category**:
  - **[Category Name]**:
    - Min: [Value]
    - Max: [Value]
    - Average: [Value]
  - **[Category Name]**:
    - Min: [Value]
    - Max: [Value]
    - Average: [Value]
- **Overall Trend**:
  - Describe whether values are increasing, decreasing, fluctuating, or stable.
- **Significant Points**:
  - Identify any peaks, lowest points, or anomalies.

#### **5. Additional Information (if applicable)**
- **Pie Chart Percentages** (if applicable):
  - List each segment and its corresponding percentage.
- **Stacked Chart Breakdown**:
  - Provide numerical values for each category in a stacked chart.
- **Annotations & Markers**:
  - Capture any additional labeled points, highlighted sections, or special markers.

---
**Final Output Structure Example (for an image with multiple charts):**  
- **Chart 1**: Bar Chart - Monthly Sales (January-December 2023)  
- **Chart 2**: Line Chart - Revenue Trend Over the Past 5 Years  
- **Chart 3**: Pie Chart - Market Share Distribution by Company  

Each chart's data should be extracted and structured in a **clear, numerical, and category-wise format**.

"""

class ExtractAgent():
    def __init__(self, system_prompt:str = None):
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = EXTRACT_PROMPT


    def run(self, query: str, image_path: str) -> str:
        image = PIL.Image.open(image_path)


        extraction = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=self.system_prompt),
            contents=[query, image]
        )

        return extraction.text.strip()
    
extract_agent = ExtractAgent()

