#!/bin/bash

# PP3 Pandas GitHub Deployment Script
# Author: George Dorochov
# Email: jordanaftermidnight@gmail.com

echo "PP3 Pandas - GitHub Deployment"
echo "=============================="

# Navigate to project directory
cd /Users/jordan_after_midnight/PP3_Pandas

# Initialize git repository
echo "Initializing git repository..."
git init

# Add all files
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: PP3 Pandas complete notebook

- Complete solutions for all 10 sections
- Comprehensive data analysis and manipulation
- Real-world datasets and examples
- Professional documentation and code structure

Author: George Dorochov
Email: jordanaftermidnight@gmail.com
Project: PP3 Pandas

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Add GitHub remote
echo "Adding GitHub remote..."
git branch -M main
git remote add origin https://github.com/jordanaftermidnight/PP3_Pandas.git

echo ""
echo "✅ Git repository initialized and committed!"
echo "📋 Next steps:"
echo "1. Create repository 'PP3_Pandas' on GitHub at https://github.com/jordanaftermidnight"
echo "2. Run: git push -u origin main"
echo "3. Or use GitHub CLI: gh repo create PP3_Pandas --public --push"
echo ""
echo "🔗 Repository will be available at: https://github.com/jordanaftermidnight/PP3_Pandas"