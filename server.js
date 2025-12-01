import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { google } from 'googleapis';
import nodemailer from 'nodemailer';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());
app.use(express.static('.')); // Serve static files from current directory

// --- Google Sheets Setup ---
const SPREADSHEET_ID = process.env.SPREADSHEET_ID;
const SHEET_NAME = 'Students';

let sheets;
let auth;

// Initialize Google Sheets API
async function initializeGoogleSheets() {
  try {
    const credentials = JSON.parse(process.env.GOOGLE_CREDENTIALS || '{}');
    auth = new google.auth.GoogleAuth({
      credentials,
      scopes: ['https://www.googleapis.com/auth/spreadsheets'],
    });

    sheets = google.sheets({ version: 'v4', auth });
    console.log('✅ Google Sheets initialized successfully');
    await ensureSheetAndHeaders();
  } catch (error) {
    console.error('❌ Failed to initialize Google Sheets:', formatError(error));
  }
}

// Ensure sheet exists and create headers if needed
async function ensureSheetAndHeaders() {
  try {
    const spreadsheet = await sheets.spreadsheets.get({
      spreadsheetId: SPREADSHEET_ID,
    });

    const sheetExists = spreadsheet.data.sheets.some(
      (s) => s.properties.title === SHEET_NAME
    );

    if (!sheetExists) {
      console.log(`ℹ️ Sheet "${SHEET_NAME}" not found. Creating it...`);
      await sheets.spreadsheets.batchUpdate({
        spreadsheetId: SPREADSHEET_ID,
        resource: {
          requests: [
            { addSheet: { properties: { title: SHEET_NAME } } },
          ],
        },
      });
      console.log(`✅ Sheet "${SHEET_NAME}" created`);
    }

    await createHeadersIfNeeded();
  } catch (error) {
    console.error('Error ensuring sheet and headers:', formatError(error));
  }
}

// Create headers if the sheet is empty
async function createHeadersIfNeeded() {
  try {
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A1:L1`,
    });

    if (!response.data.values || response.data.values.length === 0) {
      const headers = [
        'Name', 'Email', 'Major', 'Year', 'Language', 'Country',
        'Personality', 'Study Style', 'Cuisine', 'Interests', 'Movies', 'Timestamp'
      ];

      await sheets.spreadsheets.values.update({
        spreadsheetId: SPREADSHEET_ID,
        range: `${SHEET_NAME}!A1:L1`,
        valueInputOption: 'RAW',
        resource: { values: [headers] },
      });

      console.log('✅ Headers created');
    }
  } catch (error) {
    console.error('Error creating headers:', formatError(error));
  }
}

function formatError(error) {
  if (error.response && error.response.data) {
    return JSON.stringify(error.response.data, null, 2);
  }
  return error.message || error.toString();
}

// --- Nodemailer Setup ---
const transporter = nodemailer.createTransport({
  service: 'gmail', // Using Gmail for sending emails (change to other service if needed)
  auth: {
    user: process.env.EMAIL_USER, // Your email address
    pass: process.env.EMAIL_PASSWORD, // Your email password or app-specific password
  },
});

// Helper function to convert percentage to star rating
function getStarRating(percentage) {
  const stars = Math.round((percentage / 100) * 5 * 2) / 2; // Round to nearest 0.5
  const fullStars = Math.floor(stars);
  const hasHalfStar = stars % 1 !== 0;
  const emptyStars = 5 - Math.ceil(stars);
  
  let starHtml = '';
  
  // Full stars
  for (let i = 0; i < fullStars; i++) {
    starHtml += '<span style="color: #fbbf24; font-size: 20px;">★</span>';
  }
  
  // Half star
  if (hasHalfStar) {
    starHtml += '<span style="color: #fbbf24; font-size: 20px;">⯨</span>';
  }
  
  // Empty stars
  for (let i = 0; i < emptyStars; i++) {
    starHtml += '<span style="color: #d1d5db; font-size: 20px;">☆</span>';
  }
  
  return starHtml;
}

// Helper function to get compatibility level and styling
function getCompatibilityLevel(percentage) {
  if (percentage >= 85) {
    return { level: 'Exceptional', emoji: '🌟', color: '#10b981' };
  } else if (percentage >= 75) {
    return { level: 'Excellent', emoji: '⭐', color: '#3b82f6' };
  } else if (percentage >= 65) {
    return { level: 'Very Good', emoji: '💫', color: '#8b5cf6' };
  } else if (percentage >= 50) {
    return { level: 'Good', emoji: '✨', color: '#f59e0b' };
  } else {
    return { level: 'Fair', emoji: '💡', color: '#6b7280' };
  }
}

// Generate varied, well-written match explanation without repeating commonalities
function generateMatchExplanation(match, matchIndex) {
  const similarityScore = match.similarity_score || 0;
  const commonalities = match.commonalities || [];
  const commonalityCount = commonalities.length;
  
  // Select explanation template based on match index for variety
  const templateIndex = matchIndex % 6;
  
  if (similarityScore >= 85) {
    const templates = [
      `This is an exceptional match! Your compatibility score of ${similarityScore}% indicates outstanding alignment across multiple areas. This partnership has great potential for collaborative success.`,
      `With a compatibility score of ${similarityScore}%, this represents one of your strongest matches. The high level of shared interests and values suggests an ideal study partnership.`,
      `An exceptional ${similarityScore}% compatibility score demonstrates remarkable synergy between your profiles. This connection shows excellent potential for productive collaboration.`,
      `This outstanding match, scoring ${similarityScore}%, indicates a highly compatible partnership. Your shared characteristics create an ideal foundation for academic collaboration.`,
      `With ${similarityScore}% compatibility, this is an exceptional connection. Your aligned interests and values suggest you'll work together seamlessly as study partners.`,
      `This is one of your strongest matches at ${similarityScore}% compatibility. The exceptional alignment in your profiles indicates great potential for a successful study partnership.`
    ];
    return templates[templateIndex];
  } else if (similarityScore >= 75) {
    const templates = [
      `Excellent compatibility at ${similarityScore}%! Your shared interests and complementary profiles make you great study partners with strong potential for mutual success.`,
      `This excellent match (${similarityScore}%) shows strong compatibility. Your aligned preferences and similar academic approach create a solid foundation for collaboration.`,
      `With ${similarityScore}% compatibility, this is an excellent match. Your shared characteristics suggest you'll complement each other well in study sessions.`,
      `An excellent ${similarityScore}% compatibility score indicates a strong partnership. Your aligned interests provide a great starting point for productive study collaboration.`,
      `This excellent match at ${similarityScore}% compatibility shows strong potential. Your shared values and interests create an ideal environment for learning together.`,
      `Excellent compatibility of ${similarityScore}% demonstrates great alignment. Your similar preferences suggest you'll work effectively together as study partners.`
    ];
    return templates[templateIndex];
  } else if (similarityScore >= 65) {
    const templates = [
      `Very good compatibility at ${similarityScore}%! Your profiles show meaningful connections that suggest a productive and enjoyable study partnership.`,
      `This very good match (${similarityScore}%) indicates solid compatibility. Your shared interests provide common ground for effective collaboration.`,
      `With ${similarityScore}% compatibility, this is a very good match. Your aligned preferences suggest you'll find value in studying together.`,
      `A very good ${similarityScore}% compatibility score shows promising alignment. Your shared characteristics indicate potential for successful study collaboration.`,
      `This very good match at ${similarityScore}% compatibility demonstrates meaningful connections. Your similar interests create opportunities for productive learning.`,
      `Very good compatibility of ${similarityScore}% indicates a solid partnership. Your aligned preferences suggest you'll complement each other well in academic settings.`
    ];
    return templates[templateIndex];
  } else if (similarityScore >= 50) {
    const templates = [
      `Good compatibility at ${similarityScore}%! While you may have different backgrounds, your profiles complement each other, offering diverse perspectives for collaborative learning.`,
      `This good match (${similarityScore}%) shows complementary strengths. Your differences can be an asset, bringing varied perspectives to your study sessions.`,
      `With ${similarityScore}% compatibility, this is a good match. Your unique perspectives can enrich each other's learning experience.`,
      `A good ${similarityScore}% compatibility score indicates complementary profiles. Your diverse backgrounds offer opportunities for mutual learning and growth.`,
      `This good match at ${similarityScore}% compatibility shows potential. While you may approach things differently, these differences can enhance your study collaboration.`,
      `Good compatibility of ${similarityScore}% suggests a complementary partnership. Your varied perspectives can create a dynamic and enriching study environment.`
    ];
    return templates[templateIndex];
  } else {
    const templates = [
      `Fair compatibility at ${similarityScore}%. While you may have different interests, you both bring unique perspectives that could lead to interesting discussions and learning opportunities.`,
      `This fair match (${similarityScore}%) represents a complementary partnership. Your different backgrounds offer opportunities to learn from each other's perspectives.`,
      `With ${similarityScore}% compatibility, this match brings together diverse viewpoints. These differences can create a rich learning environment.`,
      `A fair ${similarityScore}% compatibility score indicates diverse profiles. Your unique characteristics can contribute to a well-rounded study experience.`,
      `This fair match at ${similarityScore}% compatibility shows complementary strengths. Your different approaches can offer fresh perspectives to each other.`,
      `Fair compatibility of ${similarityScore}% suggests a partnership with diverse perspectives. These differences can enhance your learning through varied viewpoints.`
    ];
    return templates[templateIndex];
  }
}

// Function to send email with match results
async function sendMatchEmail(studentData, matches) {
  console.log(`📧 Preparing email for ${studentData.email}`);
  console.log(`📊 Number of matches: ${matches ? matches.length : 0}`);
  console.log(`📊 Matches type: ${Array.isArray(matches) ? 'array' : typeof matches}`);
  console.log(`📊 Matches value:`, matches);
  
  // Ensure matches is an array
  if (!matches || !Array.isArray(matches)) {
    console.error(`❌ Invalid matches parameter: expected array, got ${typeof matches}`);
    matches = [];
  }
  
  // Build HTML email with professional design and star ratings
  let matchCards = '';
  
  if (matches && matches.length > 0) {
    console.log(`✓ Found ${matches.length} matches for email`);
    matchCards = matches.map((match, index) => {
      const percentage = match.similarity_score || 0;
      const stars = getStarRating(percentage);
      const compat = getCompatibilityLevel(percentage);
      
      // Generate personalized match explanation without repeating commonalities
      // The commonalities are already shown in the tags section below, so we just need a unique explanation
      const matchReason = generateMatchExplanation(match, index);
      
      return `
        <div class="match-card" style="background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%); border-radius: 16px; padding: 28px; margin-bottom: 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); border: 2px solid #e5e7eb; transition: all 0.3s;">
          <!-- Match Header -->
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 2px solid #f3f4f6;">
            <div style="flex: 1;">
              <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: linear-gradient(135deg, ${compat.color} 0%, ${compat.color}dd 100%); color: white; width: 32px; height: 32px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px;">${index + 1}</span>
                <h2 class="match-title" style="margin: 0; color: #111827; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                  ${match.name}
                </h2>
              </div>
              <div style="margin-left: 40px;">
                <div style="margin-bottom: 4px;">${stars}</div>
                <span style="color: ${compat.color}; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                  ${compat.emoji} ${compat.level} Match
                </span>
              </div>
            </div>
          </div>
          
          <!-- Contact Info -->
          <div class="contact-box" style="background: #f8fafc; padding: 16px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid ${compat.color};">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
              <svg width="16" height="16" fill="${compat.color}" viewBox="0 0 20 20">
                <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z"/>
                <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z"/>
              </svg>
              <span class="contact-label" style="font-weight: 600; color: #475569; font-size: 13px;">Contact Email</span>
            </div>
            <a href="mailto:${match.email}" style="color: ${compat.color}; text-decoration: none; font-size: 15px; font-weight: 500; display: block; margin-left: 24px;">
              ${match.email || 'N/A'}
            </a>
          </div>

          <!-- Why You Match -->
          <div class="why-match-box" style="background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%); padding: 18px; border-radius: 12px; border: 1px solid #bfdbfe;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
              <span style="font-size: 18px;">🤝</span>
              <h3 class="why-match-title" style="margin: 0; color: #1e40af; font-size: 15px; font-weight: 700;">Why You Match</h3>
            </div>
            <p class="why-match-text" style="margin: 0; color: #1e3a8a; font-size: 14px; line-height: 1.7; font-weight: 400;">
              ${matchReason}
            </p>
          </div>

          <!-- Commonalities Tags -->
          ${match.commonalities && match.commonalities.length > 0 ? `
          <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #e5e7eb;">
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
              ${match.commonalities.slice(0, 4).map(common => `
                <span style="background: ${compat.color}15; color: ${compat.color}; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid ${compat.color}30;">
                  ${common.length > 30 ? common.substring(0, 30) + '...' : common}
                </span>
              `).join('')}
            </div>
          </div>
          ` : ''}
        </div>
      `;
    }).join('');
  } else {
    matchCards = `
      <div class="no-matches-box" style="background: #fffbeb; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
        <p class="no-matches-text" style="margin: 0; color: #92400e; font-size: 14px; line-height: 1.6;">
          We're still growing our community! Check back soon as more students join, and we'll find you the perfect study partners.
        </p>
      </div>
    `;
  }

  const htmlBody = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta name="color-scheme" content="light dark">
      <meta name="supported-color-schemes" content="light dark">
      <style>
        /* Dark mode support for email clients */
        :root {
          color-scheme: light dark;
          supported-color-schemes: light dark;
        }
        
        @media (prefers-color-scheme: dark) {
          .email-body {
            background-color: #1a1a1a !important;
          }
          .email-container {
            background-color: #1a1a1a !important;
          }
          .content-box {
            background-color: #2d2d2d !important;
            color: #e5e5e5 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
          }
          .text-primary {
            color: #e5e5e5 !important;
          }
          .text-secondary {
            color: #a3a3a3 !important;
          }
          .cta-box {
            background-color: #262626 !important;
          }
          .footer-text {
            color: #737373 !important;
          }
          .match-card {
            background: linear-gradient(135deg, #2d2d2d 0%, #262626 100%) !important;
            border-color: #404040 !important;
          }
          .match-title {
            color: #f5f5f5 !important;
          }
          .contact-box {
            background-color: #262626 !important;
          }
          .contact-label {
            color: #a3a3a3 !important;
          }
          .why-match-box {
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
            border-color: #1e40af !important;
          }
          .why-match-title {
            color: #93c5fd !important;
          }
          .why-match-text {
            color: #dbeafe !important;
          }
          .no-matches-box {
            background-color: #422006 !important;
            border-color: #f59e0b !important;
          }
          .no-matches-text {
            color: #fcd34d !important;
          }
        }
      </style>
    </head>
    <body class="email-body" style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; background: #f1f5f9;">
      <div class="email-container" style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; border-radius: 12px 12px 0 0; text-align: center;">
          <div style="background: rgba(255,255,255,0.2); width: 80px; height: 80px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 20px; border: 2px solid rgba(255,255,255,0.3);">
            <svg style="width: 40px; height: 40px; fill: white;" viewBox="0 0 24 24">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
              <circle cx="9" cy="7" r="4"></circle>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
          </div>
          <h1 style="color: white; font-size: 28px; font-weight: 700; margin: 0 0 10px 0;">Hello ${studentData.name}!</h1>
          <p style="color: rgba(255,255,255,0.95); font-size: 16px; margin: 0;">Your study partner matches are ready</p>
        </div>

        <!-- Content -->
        <div class="content-box" style="background: white; padding: 30px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
          <p class="text-secondary" style="color: #475569; font-size: 15px; line-height: 1.7; margin: 0 0 30px 0;">
            Based on your profile, we've found you <strong>${matches ? matches.length : 0} ${matches && matches.length === 1 ? 'perfect match' : 'compatible study partners'}</strong>. 
            Here's who you should connect with:
          </p>

          ${matchCards}

          <!-- CTA -->
          <div class="cta-box" style="background: #f8fafc; padding: 20px; border-radius: 8px; margin-top: 30px; text-align: center;">
            <p class="text-secondary" style="margin: 0 0 16px 0; color: #64748b; font-size: 14px;">Ready to start studying together?</p>
            <p class="text-primary" style="margin: 0; color: #1e293b; font-size: 14px; line-height: 1.6;">
              Reach out to your matches using the email addresses provided above. Good luck with your studies!
            </p>
          </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; padding: 20px 0;">
          <p class="footer-text" style="margin: 0 0 8px 0; color: #94a3b8; font-size: 13px;">Studentie: AI Matcher</p>
          <p class="footer-text" style="margin: 0; color: #94a3b8; font-size: 13px;">Powered by machine learning to connect students</p>
        </div>
      </div>
    </body>
    </html>
  `;

  // Plain text fallback with star ratings
  const textBody = `
Hello ${studentData.name}!

${matches && matches.length > 0 ? `We've found your top ${matches.length} study partner matches!` : `Welcome! Your profile has been saved successfully.`}

${matches && matches.length > 0 ? matches.map((match, index) => {
  const percentage = match.similarity_score || 0;
  const stars = Math.round((percentage / 100) * 5);
  const starText = '★'.repeat(stars) + '☆'.repeat(5 - stars);
  const compat = getCompatibilityLevel(percentage);
  
  // Generate unique explanation without repeating commonalities
  const matchExplanation = generateMatchExplanation(match, index);
  
  const commonalitiesText = match.commonalities && match.commonalities.length > 0
    ? match.commonalities.slice(0, 4).join(' • ')
    : 'Could complement each other well';
  
  return `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${index + 1}. ${match.name}
${starText} ${compat.emoji} ${compat.level} Match

📧 Email: ${match.email || 'N/A'}

🤝 Why You Match:
${matchExplanation}

Common Interests: ${commonalitiesText}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;
}).join('\n\n') : 'We\'re still growing our community! Check back soon as more students join, and we\'ll find you the perfect study partners.'}

🚀 Ready to Connect?
Reach out to your matches using their email addresses above. Start a conversation, form a study group, or meet for coffee. Great partnerships start with a simple "hello"!

💡 Pro Tips:
• Send a friendly introduction email mentioning your shared interests
• Suggest a virtual or in-person study session
• Share your class schedule to find common study times

Best regards,
Student AI Matcher Team
  `;

  // Dynamic subject line based on match count
  const matchCount = matches ? matches.length : 0;
  const subjectLine = matchCount > 0
    ? `🎯 ${studentData.name}, Your ${matchCount} Perfect Study Partner${matchCount === 1 ? '' : 's'} Await!`
    : `🌟 Welcome ${studentData.name}! Your Profile is Ready`;

  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: studentData.email,
    subject: subjectLine,
    text: textBody,
    html: htmlBody,
  };

  // Validate email configuration before attempting to send
  if (!process.env.EMAIL_USER) {
    console.error('❌ EMAIL_USER is not set in .env file');
    return false;
  }
  
  if (!process.env.EMAIL_PASSWORD) {
    console.error('❌ EMAIL_PASSWORD is not set in .env file');
    return false;
  }
  
  if (process.env.EMAIL_PASSWORD.length !== 16) {
    console.error(`❌ EMAIL_PASSWORD is ${process.env.EMAIL_PASSWORD.length} characters, but Gmail App Passwords must be 16 characters`);
    console.error('   Get a new App Password at: https://myaccount.google.com/apppasswords');
    return false;
  }

  try {
    console.log('📧 Attempting to send email...');
    console.log('   From:', process.env.EMAIL_USER);
    console.log('   To:', studentData.email);
    console.log('   Subject:', subjectLine);
    console.log('   Matches:', matches ? matches.length : 0);
    
    // Verify transporter is configured
    if (!transporter) {
      console.error('❌ Email transporter is not configured');
      return false;
    }
    
    const info = await transporter.sendMail(mailOptions);
    console.log('✅ Match email sent successfully!');
    console.log('✅ Message ID:', info.messageId);
    console.log('✅ Email sent to:', studentData.email);
    console.log('✅ Response:', info.response);
    return true;
  } catch (error) {
    console.error('❌ Error sending match email:');
    console.error('   Error message:', error.message);
    console.error('   Error code:', error.code);
    console.error('   Error command:', error.command);
    console.error('   Error response:', error.response);
    console.error('   Full error:', JSON.stringify(error, null, 2));
    
    // Provide helpful error messages
    if (error.code === 'EAUTH') {
      console.error('   ⚠️ Authentication failed!');
      console.error('   - Check that EMAIL_USER is correct');
      console.error('   - Check that EMAIL_PASSWORD is a valid 16-character Gmail App Password');
      console.error('   - Get App Password at: https://myaccount.google.com/apppasswords');
    } else if (error.code === 'ECONNECTION') {
      console.error('   ⚠️ Connection failed!');
      console.error('   - Check your internet connection');
      console.error('   - Check that Gmail SMTP is accessible');
    } else if (error.code === 'ETIMEDOUT') {
      console.error('   ⚠️ Connection timed out!');
      console.error('   - Check your internet connection');
    }
    
    return false;
  }
}

// --- API Routes ---

// Serve the main HTML file
app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'student-matcher.html'));
});

// Helper function to safely parse JSON or Python-style lists
function safeParseArray(value) {
  // Handle null, undefined, or empty string
  if (value == null || value === '') {
    return [];
  }
  
  // If it's already an array, return it
  if (Array.isArray(value)) {
    return value;
  }
  
  // Convert to string if not already
  const strValue = String(value).trim();
  
  // Try to parse as JSON first (handles valid JSON like ["Chinese", "Italian"])
  try {
    const parsed = JSON.parse(strValue);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    // If parsed value is not an array, wrap it
    return [parsed];
  } catch (jsonError) {
    // JSON parse failed, try to handle Python-style lists or other formats
    try {
      // Handle Python-style lists with single quotes: ['Chinese', 'Italian']
      if (strValue.startsWith('[') && strValue.endsWith(']')) {
        // Use regex to replace single quotes around words, but be careful with nested quotes
        // This regex matches: 'word' but not 'word'word'
        let cleaned = strValue
          .replace(/'/g, '"')  // Replace all single quotes with double quotes
          .replace(/""/g, '"'); // Fix double double quotes if any
        
        try {
          return JSON.parse(cleaned);
        } catch (e) {
          // If that still fails, try manual parsing
          // Remove brackets and split by comma
          const inner = strValue.slice(1, -1).trim();
          if (!inner) return [];
          
          // Split by comma, handling quoted strings
          const items = [];
          let current = '';
          let inQuotes = false;
          let quoteChar = null;
          
          for (let i = 0; i < inner.length; i++) {
            const char = inner[i];
            if ((char === '"' || char === "'") && (i === 0 || inner[i-1] !== '\\')) {
              if (!inQuotes) {
                inQuotes = true;
                quoteChar = char;
              } else if (char === quoteChar) {
                inQuotes = false;
                quoteChar = null;
              }
            } else if (char === ',' && !inQuotes) {
              if (current.trim()) {
                items.push(current.trim().replace(/^['"]|['"]$/g, ''));
              }
              current = '';
              continue;
            }
            current += char;
          }
          
          if (current.trim()) {
            items.push(current.trim().replace(/^['"]|['"]$/g, ''));
          }
          
          return items.filter(item => item.length > 0);
        }
      }
      
      // Handle comma-separated string: "Chinese, Italian"
      if (strValue.includes(',')) {
        return strValue.split(',').map(item => item.trim().replace(/^['"]|['"]$/g, '')).filter(item => item.length > 0);
      }
      
      // Single value, return as array
      const cleaned = strValue.replace(/^['"]|['"]$/g, '');
      return cleaned ? [cleaned] : [];
    } catch (parseError) {
      console.warn(`⚠️ Failed to parse array value: "${value}"`, parseError.message);
      // Return empty array if all parsing fails
      return [];
    }
  }
}

// GET all students
app.get('/api/students', async (req, res) => {
  try {
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A2:L`,
    });

    const rows = response.data.values || [];
    console.log(`📊 Loaded ${rows.length} rows from Google Sheets`);
    
    const students = rows.map((row, index) => {
      try {
        return {
          name: row[0] || '',
          email: row[1] || '',
          major: row[2] || '',
          year: row[3] || '',
          language: row[4] || '',
          country: row[5] || '',
          personality: row[6] || '',
          studyStyle: row[7] || '',
          cuisine: safeParseArray(row[8]),
          interests: safeParseArray(row[9]),
          movies: safeParseArray(row[10]),
          timestamp: row[11] || '',
        };
      } catch (rowError) {
        console.error(`⚠️ Error parsing row ${index + 2}:`, rowError.message);
        console.error(`   Row data:`, row);
        // Return a valid object with empty arrays for problematic fields
        return {
          name: row[0] || '',
          email: row[1] || '',
          major: row[2] || '',
          year: row[3] || '',
          language: row[4] || '',
          country: row[5] || '',
          personality: row[6] || '',
          studyStyle: row[7] || '',
          cuisine: [],
          interests: [],
          movies: [],
          timestamp: row[11] || '',
        };
      }
    });

    console.log(`✅ Successfully parsed ${students.length} students`);
    res.json(students);
  } catch (error) {
    console.error('❌ Error fetching students:', formatError(error));
    console.error('❌ Error details:', error.message);
    res.status(500).json({ error: 'Failed to fetch students', details: error.message });
  }
});

// POST new student → Google Sheets + trigger Flask AI matcher
app.post('/api/students', async (req, res) => {
  try {
    const studentData = req.body;
    const required = ['name', 'email', 'major', 'year', 'language', 'country', 'personality', 'studyStyle'];

    for (const field of required) {
      if (!studentData[field]) {
        return res.status(400).json({ error: `Missing required field: ${field}` });
      }
    }

    const newRow = [
      studentData.name,
      studentData.email,
      studentData.major,
      studentData.year,
      studentData.language,
      studentData.country,
      studentData.personality,
      studentData.studyStyle,
      JSON.stringify(studentData.cuisine || []),
      JSON.stringify(studentData.interests || []),
      JSON.stringify(studentData.movies || []),
      new Date().toISOString(),
    ];

    // Append new student to Google Sheets
    let sheetsResult;
    try {
      sheetsResult = await sheets.spreadsheets.values.append({
        spreadsheetId: SPREADSHEET_ID,
        range: `${SHEET_NAME}!A:L`,
        valueInputOption: 'RAW',
        resource: { values: [newRow] },
      });
      console.log(`📄 Row appended to "${SHEET_NAME}":`, newRow);
      console.log(`✅ Student saved to Google Sheets successfully`);
    } catch (sheetsError) {
      console.error('❌ Error writing to Google Sheets:', sheetsError);
      return res.status(500).json({ 
        error: 'Failed to save data to Google Sheets',
        success: false 
      });
    }

    // Wait a moment for Google Sheets API to update before calling Flask
    // This ensures the new student is available when Flask loads data
    // Increased to 3 seconds to ensure Google Sheets API has time to update
    console.log('⏳ Waiting 3 seconds for Google Sheets to update...');
    await new Promise(resolve => setTimeout(resolve, 3000));

    // --- Trigger Flask backend for AI matching ---
    try {
      const encodedName = encodeURIComponent(studentData.name);
      // Use localhost consistently (same as Flask uses for Node.js)
      const flaskUrl = `http://localhost:5000/api/match?name=${encodedName}`;
      console.log(`⚙️ Triggering Flask backend for match computation (user=${studentData.name})...`);
      console.log(`   Flask URL: ${flaskUrl}`);

      // Create AbortController for timeout (more compatible than AbortSignal.timeout)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(flaskUrl, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ Flask backend error (${response.status}):`, errorText);
        
        // Try to parse error response
        let errorData;
        try {
          errorData = JSON.parse(errorText);
        } catch (e) {
          errorData = { error: errorText };
        }
        
        throw new Error(`Flask backend responded with ${response.status}: ${errorData.error || errorText}`);
      }

      const matchData = await response.json();
      console.log('✅ Match results from Flask:', JSON.stringify(matchData, null, 2));

      // Check if Flask returned success
      if (matchData.success === false || matchData.error) {
        console.error('⚠️ Flask returned error:', matchData.error || 'Unknown error');
        throw new Error(matchData.error || 'Flask matching failed');
      }

      // Extract matches from response - handle both 'matches' and 'all_matches' fields
      let matches = [];
      
      // Debug: Log the structure of matchData
      console.log('🔍 Debug: matchData structure:', {
        hasMatches: !!matchData.matches,
        matchesType: typeof matchData.matches,
        matchesIsArray: Array.isArray(matchData.matches),
        matchesLength: matchData.matches ? (Array.isArray(matchData.matches) ? matchData.matches.length : 'not array') : 'undefined',
        hasAllMatches: !!matchData.all_matches,
        allMatchesType: typeof matchData.all_matches,
        totalMatches: matchData.total_matches,
        studentName: matchData.student_name,
        keys: Object.keys(matchData)
      });
      
      // Priority 1: Check if 'matches' field exists and is an array (preferred format)
      if (matchData.matches !== undefined && matchData.matches !== null) {
        if (Array.isArray(matchData.matches)) {
          matches = matchData.matches;
          console.log(`✅ Found ${matches.length} matches in matchData.matches array`);
        } else {
          console.log(`⚠️ matchData.matches exists but is not an array: ${typeof matchData.matches}`);
        }
      }
      
      // Priority 2: If no matches found, check 'all_matches' object
      if (matches.length === 0 && matchData.all_matches && typeof matchData.all_matches === 'object') {
        const studentName = studentData.name;
        console.log(`🔍 Looking for matches in all_matches for student: "${studentName}"`);
        console.log(`   Available keys in all_matches: ${Object.keys(matchData.all_matches).slice(0, 10).join(', ')}`);
        
        // Try exact match first
        if (matchData.all_matches[studentName]) {
          matches = matchData.all_matches[studentName];
          console.log(`✅ Found ${matches.length} matches using exact name match`);
        } else {
          // Try case-insensitive match
          for (const key in matchData.all_matches) {
            if (key.toLowerCase().trim() === studentName.toLowerCase().trim()) {
              matches = matchData.all_matches[key];
              console.log(`✅ Found ${matches.length} matches using case-insensitive match (key: "${key}")`);
              break;
            }
          }
        }
      }
      
      // Final check: Ensure matches is an array
      if (!Array.isArray(matches)) {
        console.error(`❌ Matches is not an array: ${typeof matches}, value: ${matches}`);
        matches = [];
      }
      
      // Log match details for debugging
      const matchCount = matches.length;
      console.log(`📧 Preparing to send email with ${matchCount} matches`);
      console.log(`📧 Student: ${studentData.name}`);
      console.log(`📧 Student email: ${studentData.email}`);
      console.log(`📧 Email from: ${process.env.EMAIL_USER || 'NOT SET'}`);
      console.log(`📧 Email password length: ${process.env.EMAIL_PASSWORD ? process.env.EMAIL_PASSWORD.length : 0} characters`);
      
      if (matchCount > 0) {
        console.log(`📋 Match names: ${matches.map(m => m.name || 'N/A').join(', ')}`);
        console.log(`📋 Match scores: ${matches.map(m => m.similarity_score || 0).join(', ')}`);
        console.log(`📋 First match structure:`, matches[0] ? Object.keys(matches[0]) : 'no matches');
        // Validate matches structure
        matches.forEach((match, idx) => {
          if (!match.name) {
            console.error(`⚠️ Match ${idx} missing name field`);
          }
          if (!match.email) {
            console.error(`⚠️ Match ${idx} missing email field`);
          }
          if (match.similarity_score === undefined || match.similarity_score === null) {
            console.error(`⚠️ Match ${idx} missing similarity_score field`);
          }
        });
      } else {
        console.log(`⚠️ No matches found for ${studentData.name}`);
        console.log(`   Flask returned: success=${matchData.success}, total_matches=${matchData.total_matches || 0}`);
        console.log(`   This could mean:`);
        console.log(`   - Student just registered (needs more students for matches)`);
        console.log(`   - Similarity scores below threshold`);
        console.log(`   - Student name mismatch between form and database`);
      }
      
      const emailSent = await sendMatchEmail(studentData, matches);
      if (emailSent) {
        console.log('✅ Email sent successfully!');
      } else {
        console.error('❌ Email sending failed!');
        console.error('   Check .env file for EMAIL_USER and EMAIL_PASSWORD');
        console.error('   EMAIL_PASSWORD must be a 16-character Gmail App Password');
        console.error('   Get App Password at: https://myaccount.google.com/apppasswords');
      }

      // ✅ Return matches to frontend
      // Use the matches we extracted above (not matchData.matches which might be undefined)
      const actualMatches = matches; // Already extracted above
      const actualMatchCount = actualMatches.length;
      
      console.log(`📊 Match Summary:`);
      console.log(`   Student: ${studentData.name}`);
      console.log(`   Matches found: ${actualMatchCount}`);
      console.log(`   Flask returned: ${matchData.total_matches || 0}`);
      console.log(`   Actual matches: ${actualMatchCount}`);
      
      const response_data = {
        success: true,
        message: 'Student added successfully',
        student: studentData,
        matches: actualMatches,
        totalMatches: actualMatchCount, // Use actual count, not Flask's count
        emailSent: emailSent || false,
      };
      
      console.log('📤 Returning to frontend:', JSON.stringify(response_data, null, 2));
      return res.json(response_data);
    } catch (flaskError) {
      console.error('❌ Failed to contact Flask backend:', flaskError.message);
      console.error('   Error type:', flaskError.name);
      console.error('   Note: Student was added to Google Sheets successfully');
      
      // Check if it's a connection error
      if (flaskError.message.includes('ECONNREFUSED') || 
          flaskError.message.includes('fetch failed') ||
          flaskError.message.includes('ECONNRESET') ||
          flaskError.name === 'AbortError' ||
          flaskError.code === 'ECONNREFUSED') {
        console.error('   ⚠️ Flask backend is not running on port 5000 or connection failed');
        console.error('   💡 Please start Flask backend: cd app_backend && python app.py');
        console.error('   💡 Make sure Flask is listening on http://localhost:5000');
      } else if (flaskError.message.includes('timeout') || flaskError.message.includes('TIMEOUT')) {
        console.error('   ⚠️ Flask backend request timed out after 30 seconds');
        console.error('   💡 Flask backend may be overloaded or not responding');
      }
      
      // Still send a "no matches" email to user
      let emailSent = false;
      try {
        console.log('📧 Sending welcome email (no matches available)...');
        emailSent = await sendMatchEmail(studentData, []);
        if (emailSent) {
          console.log('✅ Welcome email sent successfully!');
        } else {
          console.error('❌ Email sending failed!');
          console.error('   Check .env file for EMAIL_USER and EMAIL_PASSWORD');
        }
      } catch (emailError) {
        console.error('❌ Error sending email:', emailError.message);
        console.error('   Full error:', emailError);
      }
      
      return res.json({
        success: true,
        message: 'Student added successfully, but matches could not be computed',
        student: studentData,
        matches: [],
        totalMatches: 0,
        emailSent: emailSent,
        warning: 'Flask backend is not available. Please check if it is running on port 5000.',
      });
    }

  } catch (error) {
    console.error('Error adding student:', error);
    res.status(500).json({ error: 'Failed to add student' });
  }
});

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 Google Sheets ID: ${SPREADSHEET_ID}`);
  initializeGoogleSheets();
});
