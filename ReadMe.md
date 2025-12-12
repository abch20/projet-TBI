# 🚀 **Business Intelligence Project – Northwind Dashboard (Power BI)**

This project is a complete **Business Intelligence (BI)** solution built using **Power BI**, **Power Query**, and **DAX**, based on the well-known **Northwind** dataset.
Its goal is to transform operational data into meaningful insights to support decision-making.

## 📊 **Project Overview**

The solution covers the full BI pipeline:

### **1️⃣ Data Extraction & Transformation (ETL)**

Using **Power Query**, raw OLTP data from Northwind (Employees, Customers, Orders, Territories, Regions…) is:

- extracted from SQL Server/Access
- cleaned (NULL handling, type conversions, normalization)
- merged into analytical tables
- enriched with surrogate keys and flags

### **2️⃣ Data Warehouse Modeling**

A **Star Schema** has been designed with:

- **Fact Table:** Fait_Commandes (orders metrics)
- **Dimension Tables:**

  - Dim_Employee
  - Dim_Client
  - Dim_Temps

This structure improves performance, readability, and advanced analysis.

### **3️⃣ DAX Measures & KPIs**

Custom DAX calculations were implemented to provide business indicators:

- Total Orders
- Delivered Orders
- Delivery Rate
- Unique Clients
- Employee & Region performance

### **4️⃣ Interactive Dashboard**

The dashboard includes:

- KPI summary cards
- Time-series analysis
- Regional performance maps
- Employee and client ranking charts
- Filters for dynamic exploration (Year, Region, Employee, Client)

### **5️⃣ Performance Optimization**

- Reduced unnecessary columns in Power Query
- Efficient DAX measures (no heavy calculated columns)
- Proper data types and relationships
- Single-direction relationships to avoid ambiguity

---

## 🎯 **Purpose of the Project**

This BI project aims to:

- demonstrate end-to-end BI development
- build a dynamic and easy-to-use dashboard
- provide deep insights into sales and delivery performance
- show best practices in ETL, modeling, DAX, and visualization

---

## 🛠️ **Technologies Used**

- **Power BI Desktop**
- **Power Query (M)**
- **DAX**
- **SQL Server / Access (Northwind)**
- **Python (optional for analysis)**
- **Git / GitHub**

---

## 📁 **Repository Structure (Suggested)**

```
/Data              → Source data (optional or description)
/PowerBI           → .pbix file
/Documentation     → PDF report, schema, screenshots
README.md          → Project description
```

---

## 📎 **Key Features**

- Clean star-schema model
- Complete ETL pipeline
- KPI-driven dashboard
- Fully interactive visual experience
- Scalable and optimized BI architecture
