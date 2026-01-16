# Quick Start Guide - Streamlit App

## 🚀 Launching the App

### Option 1: Use the Launcher Script (Recommended)
**Windows:**
```bash
run_streamlit.bat
```

**Linux/Mac:**
```bash
chmod +x run_streamlit.sh
./run_streamlit.sh
```

**Or use Python directly:**
```bash
python launch_app.py
```

The launcher will:
- ✅ Check if Streamlit is installed
- ✅ Verify app.py exists
- ✅ Automatically open your browser
- ✅ Start the Streamlit server

### Option 2: Manual Launch
```bash
python -m streamlit run app.py
```

Then manually open: http://localhost:8501

## 📊 Features

### Progress Tracking
- **Stage 1**: Progress bar shows calculation of required principal for each retirement age
- **Stage 2**: Progress bar shows accumulation simulation progress (X/Y simulations)

### Interactive Configuration
All parameters are configurable via the sidebar:
- Basic parameters (age, portfolio, income, spending)
- Social Security settings
- Simulation settings (num_outer, num_nested, success target)
- Retirement age range
- Model selection (Block Bootstrap vs Parametric)
- GKOS earnings parameters
- Utility parameters
- Withdrawal strategy
- Advanced settings

### Results Display
- Required principal table by retirement age
- Retirement age statistics (median, percentiles, probabilities)
- Utility metrics (if enabled)
- Amortization statistics (if enabled)
- Interactive visualizations
- CSV download options

## 💡 Tips

1. **Start Small**: Use `num_outer=100` and `num_nested=50` for quick tests
2. **Check Runtime**: The app shows estimated runtime before running
3. **Clear Results**: Use "Clear Results" button to start fresh
4. **Progress Bars**: Watch the progress bars to see simulation status
5. **Download Results**: Use download buttons to save CSV files

## 🔧 Troubleshooting

**Streamlit not found:**
```bash
pip install streamlit
```

**App won't open in browser:**
- Manually go to: http://localhost:8501
- Check if port 8501 is already in use

**Import errors:**
- Make sure all dependencies are installed
- Check that `LIFECYCLE MODEL` directory exists
- Verify Python path is correct

## 📝 Notes

- Progress bars update in real-time during simulation
- The app automatically suppresses tqdm output for clean Streamlit interface
- All results are stored in session state until cleared
- Large simulations may take several minutes - watch the progress bars!

