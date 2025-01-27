import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function PrivacyPage() {
  return (
    <>
      <Header />
      <main className="overflow-hidden">
        <div className="relative min-h-screen">
          {/* Background effects */}
          <div className="gradient-blur absolute inset-0 -z-20" />
          <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
            <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#1E3A8A] to-[#3B82F6] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
          </div>

          <div className="mx-auto max-w-7xl px-6 pb-24 pt-32 sm:pt-40 lg:px-8 relative z-10">
            <div className="mx-auto max-w-4xl">
              <div className="container mx-auto px-4 py-8">
                <div className="prose prose-invert mx-auto max-w-4xl prose-ul:list-disc prose-ul:pl-5 prose-ol:list-decimal prose-ol:pl-5">
                  <h1 className="mb-8 text-4xl font-bold">Privacy Policy</h1>

                  <h2 className="text-8 mb-4 mt-8 font-bold">Effective Date: December 11th, 2024</h2>

                  <p>At Pr0j3c7t0dd Ltd. (UK Companies House Number - 14080231), we take your privacy seriously. This Privacy Policy outlines how we collect, use, disclose, and protect your personal data when you use our audio recording software and associated services (the &quot;Services&quot;). By accessing or using the Services, you consent to the practices described in this Privacy Policy.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">1. Definitions</h2>
                  <ul>
                    <li><strong>Services</strong>: Refers to all software, tools, and associated features provided by Pr0j3c7t0dd Ltd., including audio recording software and the company website.</li>
                    <li><strong>Personal Data</strong>: Any information relating to an identified or identifiable individual.</li>
                    <li><strong>User-Generated Content</strong>: Any content uploaded, shared, or created by users through the Services, including but not limited to audio files and transcripts.</li>
                    <li><strong>Cookie</strong>: A small text file stored on a user&apos;s device by a website to track preferences and activity.</li>
                  </ul>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">2. Scope of the Privacy Policy</h2>
                  <p>This Privacy Policy covers the collection, use, and disclosure of personal data that we gather through our website and the Services. Personal data refers to information that can identify or relates to an individual.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">3. Information We Collect</h2>
                  <p>Pr0j3c7t0dd Ltd. does not collect any user information through our software. The information collected through the software (audio recordings, transcripts, login information, etc.) is the user&apos;s sole responsibility to handle and manage according to the laws and regulations in the user&apos;s jurisdiction.</p>

                  <p>While our software does not collect user information, Pr0j3c7t0dd Ltd. may collect the following types of information through our website:</p>

                  <ol>
                    <li><strong>Profile or Contact Data:</strong>
                      <ul>
                        <li>Name</li>
                        <li>Email address</li>
                      </ul>
                    </li>

                    <li><strong>Device and Usage Data:</strong>
                      <ul>
                        <li>IP address</li>
                        <li>Device and browser information</li>
                        <li>Logs of interactions with our website</li>
                      </ul>
                    </li>

                    <li><strong>Cookies and Similar Technologies:</strong>
                      <ul>
                        <li>Information on user preferences and activity patterns</li>
                      </ul>
                    </li>

                    <li><strong>Social Login Information:</strong>
                      <ul>
                        <li>Details from services like Google OAuth when used to log in to our website</li>
                      </ul>
                    </li>
                  </ol>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">4. Cookie Policy</h2>
                  <p>We use cookies to improve user experience, analyze website usage, and support the functionality of the Services. By using our website, you consent to the use of cookies as described in this policy.</p>

                  <h3 className="mb-3 mt-6 text-xl font-bold">Types of Cookies We Use:</h3>
                  <ul>
                    <li><strong>Essential Cookies</strong>: Required for the website to function.</li>
                    <li><strong>Analytics Cookies</strong>: Help us understand user behavior and improve the website.</li>
                    <li><strong>Preference Cookies</strong>: Remember user settings and preferences.</li>
                  </ul>

                  <p>You can manage or disable cookies through your browser settings. Please note that disabling cookies may impact the functionality of the Services.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">5. How We Use Your Information</h2>
                  <p>We use personal data collected through our website to:</p>
                  <ul>
                    <li>Provide and enhance the Services.</li>
                    <li>Communicate with you about updates, features, and support.</li>
                    <li>Improve website functionality and user experience.</li>
                    <li>Comply with legal obligations and enforce terms.</li>
                  </ul>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">6. Sharing and Disclosure of Data</h2>
                  <p>We do not sell your personal data. However, we may share data collected from the website with:</p>
                  <ul>
                    <li><strong>Service Providers:</strong>
                      <ul>
                        <li>Hosting providers</li>
                        <li>Analytics tools</li>
                      </ul>
                    </li>
                    <li><strong>Legal Authorities:</strong>
                      <ul>
                        <li>To comply with legal obligations or protect rights</li>
                      </ul>
                    </li>
                    <li><strong>Third-Party Tools:</strong>
                      <ul>
                        <li>Some features of the Services rely on third-party tools or integrations. These tools collect and process data under their own privacy policies, which we encourage you to review independently. Pr0j3c7t0dd Ltd. is not liable for data handling practices by these third parties.</li>
                      </ul>
                    </li>
                  </ul>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">7. Data Security</h2>
                  <p>We implement industry-standard security measures to protect data collected through our website. However, no method of transmission or storage is completely secure, and we cannot guarantee absolute security.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">8. Data Retention</h2>
                  <p>Personal data collected through our website is retained only as long as necessary to fulfill the purposes outlined in this Privacy Policy or as required by law. Anonymized or aggregated data may be retained indefinitely for analytics or service improvement purposes.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">9. User Responsibility Disclaimer</h2>
                  <p>Pr0j3c7t0dd Ltd. provides tools for data processing but does not oversee or control how users manage or utilize the software. Users are solely responsible for ensuring their use complies with all applicable laws, including but not limited to obtaining necessary consents and adhering to privacy regulations in their jurisdiction, especially regarding the collection and use of recorded audio and derived transcripts.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">10. Children&apos;s Privacy</h2>
                  <p>The Services are not directed at individuals under 16 years of age. If we become aware of any data collected from individuals under 16, we will take immediate steps to delete it.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">11. Cross-Border Data Transfers</h2>
                  <p>If personal data is transferred outside the UK or EU, we ensure adequate safeguards are in place, such as adherence to Standard Contractual Clauses (SCCs) or equivalent protections as required under GDPR.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">12. Data Breach Policy</h2>
                  <p>In the event of a data breach involving personal data, Pr0j3c7t0dd Ltd. will notify affected users and relevant regulatory authorities within 72 hours, as required under GDPR.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">13. Your Rights</h2>
                  <p>Under the UK GDPR, you have the following rights:</p>
                  <ul>
                    <li>Access your data</li>
                    <li>Rectify inaccuracies</li>
                    <li>Request erasure of your data</li>
                    <li>Object to or restrict data processing</li>
                    <li>Request a copy of your personal data in a machine-readable format (data portability)</li>
                  </ul>

                  <p>To exercise these rights, please contact us at <a href="mailto:legal@pr0j3c7t0dd.com">legal@pr0j3c7t0dd.com</a>.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">14. Acceptable Use Policy</h2>
                  <h3 className="mb-3 mt-6 text-xl font-bold">Prohibited Activities</h3>
                  <p>Users of the Services must adhere to the following acceptable use guidelines:</p>
                  <ul>
                    <li><strong>Illegal Activities:</strong> You may not use the Services to engage in or promote illegal activities.</li>
                    <li><strong>Harmful or Offensive Content:</strong> Uploading or sharing unlawful, harmful, offensive, defamatory, or fraudulent content is prohibited.</li>
                    <li><strong>Intellectual Property Violations:</strong> Misusing the Services to infringe on intellectual property rights, including copyright, trademark, or patent violations.</li>
                    <li><strong>Automated Tools:</strong> Using automated tools or scripts, such as bots, to access or interact with the Services without authorization.</li>
                    <li><strong>Tampering with Software:</strong> Attempting to reverse-engineer, decompile, or tamper with any part of the Services.</li>
                  </ul>

                  <p>Violation of these guidelines may result in account suspension, termination, or legal action.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">15. Data Protection Officer Information</h2>
                  <h3 className="mb-3 mt-6 text-xl font-bold">15.1. DPO Appointment and Role</h3>
                  <p>Pr0j3c7t0dd Ltd. has appointed a Data Protection Officer (DPO) to oversee compliance with data protection laws and regulations, including but not limited to the UK GDPR and Data Protection Act 2018. The DPO serves as the primary point of contact for data protection matters and operates independently to ensure the protection of your personal data.</p>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.2. DPO Responsibilities</h3>
                  <p>Our DPO is responsible for:</p>
                  <ul>
                    <li>Monitoring compliance with applicable data protection laws and regulations</li>
                    <li>Advising on data protection impact assessments (DPIAs)</li>
                    <li>Serving as the primary contact point for supervisory authorities</li>
                    <li>Managing internal data protection activities</li>
                    <li>Training staff on data protection requirements</li>
                    <li>Conducting regular audits to ensure compliance</li>
                    <li>Providing guidance on data protection by design and default</li>
                    <li>Maintaining records of processing activities</li>
                  </ul>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.3. Contact Information</h3>
                  <p><strong>Data Protection Officer</strong><br />
                    Pr0j3c7t0dd Ltd.<br />
                    54 Stockbridge Road<br />
                    Winchester, Hampshire<br />
                    SO22 6RL<br />
                    United Kingdom<br />
                    Email: <a href="mailto:dpo@pr0j3c7t0dd.com">dpo@pr0j3c7t0dd.com</a></p>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.4. When to Contact the DPO</h3>
                  <p>You should contact our DPO if:</p>
                  <ul>
                    <li>You wish to exercise your data protection rights</li>
                    <li>You have questions about how we process your personal data</li>
                    <li>You want to report a potential data breach</li>
                    <li>You have concerns about our data protection practices</li>
                    <li>You need guidance on data protection compliance</li>
                    <li>You want to submit a complaint about our data handling</li>
                  </ul>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.5. Response Times</h3>
                  <p>Our DPO will acknowledge receipt of your inquiry within 2 business days and provide a substantive response within:</p>
                  <ul>
                    <li>30 calendar days for general inquiries and rights requests</li>
                    <li>72 hours for potential data breach reports</li>
                    <li>5 business days for urgent compliance matters</li>
                  </ul>

                  <p>These timeframes may be extended by up to two months for complex requests, in which case you will be notified within the initial response period.</p>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.6. Regulatory Reporting</h3>
                  <p>Our DPO maintains direct contact with the Information Commissioner&apos;s Office (ICO) and will:</p>
                  <ul>
                    <li>Report notifiable breaches within 72 hours</li>
                    <li>Conduct regular compliance assessments</li>
                    <li>Maintain required documentation</li>
                    <li>Respond to regulatory inquiries</li>
                  </ul>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.7. Independence and Authority</h3>
                  <p>The DPO:</p>
                  <ul>
                    <li>Reports directly to the highest level of management</li>
                    <li>Operates independently without receiving instructions regarding their tasks</li>
                    <li>Cannot be dismissed or penalized for performing their duties</li>
                    <li>Is provided with adequate resources to fulfill their role</li>
                    <li>Has access to all necessary information and processing operations</li>
                  </ul>

                  <h3 className="mb-3 mt-6 text-xl font-bold">15.8. Regular Updates</h3>
                  <p>The DPO provides regular updates on:</p>
                  <ul>
                    <li>Changes in data protection laws and regulations</li>
                    <li>Internal compliance status</li>
                    <li>Recommended improvements to data protection measures</li>
                    <li>Training requirements and completion</li>
                    <li>Audit findings and remediation efforts</li>
                  </ul>

                  <p>This section shall be updated as needed to reflect any changes in DPO appointment, contact information, or responsibilities. All updates will be communicated through appropriate channels and reflected in the latest version of this Privacy Policy.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">16. Contact Information</h2>
                  <p>For inquiries or complaints about this Privacy Policy, contact us at:</p>
                  <ul>
                    <li>Email: <a href="mailto:legal@pr0j3c7t0dd.com">legal@pr0j3c7t0dd.com</a></li>
                    <li>Address: 54 Stockbridge Road, Winchester, Hampshire, SO22 6RL, UK</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  );
}
