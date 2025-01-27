import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function TermsAndConditions() {
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

          <div className="relative z-10 mx-auto max-w-7xl px-6 pb-24 pt-32 sm:pt-40 lg:px-8">
            <div className="mx-auto max-w-4xl">
              <div className="container mx-auto px-4 py-8">
                <div className="prose prose-invert mx-auto max-w-4xl prose-ol:list-decimal prose-ol:pl-5 prose-ul:list-disc prose-ul:pl-5">
                  <h1 className="mb-8 text-4xl font-bold">Terms and Conditions</h1>

                  <h2 className="text-l mb-4 mt-8 font-bold">Effective Date: December 11th, 2024</h2>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">1. Acceptance of Terms</h2>
                  <p>By using our Services, you agree to these Terms and Conditions. If you do not accept these terms, you are not authorized to use the Services.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">2. User Obligations</h2>
                  <h3 className="mb-3 mt-6 text-xl font-bold">a. Compliance with Laws</h3>
                  <p>You are solely responsible for ensuring your use of the Services complies with applicable laws, including obtaining necessary consents for audio recordings. In the UK, this includes:</p>
                  <ul>
                    <li>Informing and gaining consent from participants before recording.</li>
                    <li>Ensuring recordings comply with privacy regulations.</li>
                  </ul>

                  <h3 className="mb-3 mt-6 text-xl font-bold">b. Prohibited Activities</h3>
                  <p>You agree not to:</p>
                  <ul>
                    <li>Use the Services for unlawful purposes.</li>
                    <li>Distribute harmful, offensive, or unlawful content.</li>
                    <li>Interfere with or disrupt the integrity of the Services.</li>
                    <li>Use automated means (e.g., bots) to access or use the Services.</li>
                    <li>Attempt to reverse-engineer, decompile, or otherwise tamper with the software.</li>
                  </ul>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">3. Ownership and Intellectual Property</h2>
                  <p>All content and software associated with the Services are owned by Pr0j3c7t0dd Ltd. or its licensors. You may not copy, distribute, or modify any content without explicit permission. The use of software with an open-source license, such as Apache 2.0, is permitted without prior approval, provided that Pr0j3c7t0dd Ltd. retains ownership and branding rights associated with the original software as per the license provisions.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">4. Indemnification</h2>
                  <p>You agree to indemnify, defend, and hold harmless Pr0j3c7t0dd Ltd., its affiliates, employees, and agents from any claims, liabilities, damages, or legal fees arising from your use of the Services, including but not limited to:</p>
                  <ul>
                    <li>Violations of applicable privacy laws.</li>
                    <li>Misuse of recordings or data.</li>
                    <li>Failure to obtain necessary consents for audio recordings.</li>
                  </ul>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">5. Limitation of Liability</h2>
                  <p>To the extent permitted by law:</p>
                  <ul>
                    <li>We are not liable for indirect, incidental, or consequential damages.</li>
                    <li>Our maximum liability for claims is limited to the greater of:
                      <ul>
                        <li>The amount paid by you for the Services in the six months preceding the claim.</li>
                        <li>£500.</li>
                      </ul>
                    </li>
                  </ul>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">6. Disclaimer of Warranties</h2>
                  <p>The Services are provided "as is" and "as available." We disclaim all warranties, including fitness for a particular purpose.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">7. Arbitration Agreement</h2>
                  <p>Any dispute arising out of or in connection with these Terms or the Services shall be resolved through binding arbitration in accordance with the rules of the Centre for Effective Dispute Resolution (CEDR). This does not affect any mandatory statutory rights under applicable laws. If arbitration is prohibited by law in your jurisdiction, disputes shall be resolved in accordance with the Governing Law section.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">8. Force Majeure</h2>
                  <p>Pr0j3c7t0dd Ltd. shall not be held liable for failure or delay in performing obligations due to events beyond its reasonable control, including but not limited to natural disasters, pandemics, cyberattacks, governmental restrictions, or interruptions in internet service.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">9. User-Generated Content Disclaimer</h2>
                  <p>Users are solely responsible for the content they upload, share, or generate through the Services. Pr0j3c7t0dd Ltd. does not review or validate the accuracy, legality, or reliability of user-generated content. By uploading content, users grant Pr0j3c7t0dd Ltd. a non-exclusive, royalty-free license to store, process, or delete such content as necessary to provide the Services.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">10. Termination</h2>
                  <p>We may suspend or terminate your access to the Services for violation of these Terms or as required by law. Termination does not absolve you of obligations under these Terms. No refunds will be issued for Services terminated due to user violations.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">11. Governing Law and Dispute Resolution</h2>
                  <p>These Terms are governed by the laws of England and Wales. Disputes will be resolved exclusively in the courts of England and Wales.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">12. Amendments</h2>
                  <p>We may modify these Terms at our discretion. Changes will be communicated through the website or other means.</p>

                  <h2 className="mb-4 mt-8 text-2xl font-bold">13. Contact Information</h2>
                  <p>For questions regarding these Terms, contact us at:</p>
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
